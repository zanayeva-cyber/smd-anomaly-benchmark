"""Сравнение методов обнаружения аномалий на Server Machine Dataset (SMD).

Методы:
  random      — случайная мера аномальности (контроль протокола оценки);
  cusum_ewma  — CUSUM по остаткам EWMA-прогноза (статистический метод);
  iforest     — Isolation Forest по вектору метрик в момент t;
  lstm        — LSTM-прогноз следующего вектора, мера = ошибка прогноза;
  lstm_ae     — LSTM-автоэнкодер окна, мера = ошибка восстановления;
  lstm_cusum  — гибрид: CUSUM по нормированным остаткам LSTM-прогноза.

Метрики:
  F1 (поточечно) и F1-PA (с point-adjust) при оптимальном («оракульном») пороге;
  эксплуатационные метрики при пороге по 99.5-му перцентилю меры на валидационной
  части обучающей выборки: доля обнаруженных аномальных интервалов, медианное
  запаздывание (мин), число ложных тревог в сутки.

Запуск: python smd_benchmark.py [--machines machine-1-1,machine-2-1] [--epochs 8]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'SMD'
OUT = ROOT / 'results'
SEED = 0
WIN = 30            # длина окна для нейросетей, мин
VAL_FRAC = 0.2      # доля обучающей выборки для выбора эксплуатационного порога
OP_Q = 0.995        # квантиль меры на валидации для эксплуатационного порога
CUSUM_K = 0.5       # параметр сноса CUSUM (в единицах СКО)
MERGE_GAP = 5       # тревоги ближе 5 мин считаются одной
POINTS_PER_DAY = 1440


def load(machine):
    rd = lambda sub: np.loadtxt(DATA / sub / f'{machine}.txt', delimiter=',', dtype=np.float32)
    return rd('train'), rd('test'), rd('test_label').astype(np.int8)


def segments(labels):
    d = np.diff(np.r_[0, labels, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))  # [start, end)


# ---------------------------------------------------------------- метрики

def best_f1(score, labels, segs, n_thr=400):
    """Лучший F1 по сетке порогов: поточечно и с point-adjust."""
    thr = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, n_thr)))
    pos = labels.astype(bool)
    n_pos = pos.sum()
    s_pos, s_neg = np.sort(score[pos]), np.sort(score[~pos])
    seg_max = np.array([score[a:b].max() for a, b in segs])
    seg_len = np.array([b - a for a, b in segs])
    tp = n_pos - np.searchsorted(s_pos, thr, side='left')
    fp = len(s_neg) - np.searchsorted(s_neg, thr, side='left')
    f1 = 2 * tp / np.maximum(2 * tp + fp + (n_pos - tp), 1)
    tp_pa = np.array([seg_len[seg_max >= t].sum() for t in thr])
    f1_pa = 2 * tp_pa / np.maximum(2 * tp_pa + fp + (n_pos - tp_pa), 1)
    return float(f1.max()), float(f1_pa.max())


def operational(score, labels, segs, thr):
    """Эксплуатационные метрики при фиксированном пороге."""
    alarm = score >= thr
    detected, delays = 0, []
    for a, b in segs:
        hit = np.flatnonzero(alarm[a:b])
        if hit.size:
            detected += 1
            delays.append(hit[0])
    # ложные тревоги: начала тревог вне аномальных интервалов, близкие тревоги объединяются
    outside = alarm & ~labels.astype(bool)
    idx = np.flatnonzero(outside)
    n_events = 0 if idx.size == 0 else 1 + int((np.diff(idx) > MERGE_GAP).sum())
    days = len(score) / POINTS_PER_DAY
    return dict(event_recall=detected / max(len(segs), 1),
                delay_median=float(np.median(delays)) if delays else np.nan,
                false_alarms_per_day=n_events / days,
                normal_time_in_alarm=float(outside.sum() / max((~labels.astype(bool)).sum(), 1)))


# ---------------------------------------------------------------- методы

def cusum(z, k=CUSUM_K):
    """Двусторонний CUSUM по нормированным остаткам каждой метрики (обнаружение сдвига
    среднего на ±2k СКО); мера = максимум статистики по метрикам и направлениям."""
    gp = np.zeros(z.shape[1], dtype=np.float64)
    gn = np.zeros(z.shape[1], dtype=np.float64)
    out = np.empty(len(z))
    for t, row in enumerate(z):
        gp = np.maximum(0.0, gp + row - k)
        gn = np.maximum(0.0, gn - row - k)
        out[t] = max(gp.max(), gn.max())
    return out


def ewma_residuals(x, alpha=0.3):
    pred = np.empty_like(x)
    m = x[0].copy()
    for t in range(len(x)):
        pred[t] = m
        m = alpha * x[t] + (1 - alpha) * m
    return x - pred


class Forecaster(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.rnn = nn.LSTM(d, h, batch_first=True)
        self.out = nn.Linear(h, d)

    def forward(self, w):                      # w: (B, WIN, d) -> следующий вектор (B, d)
        _, (hn, _) = self.rnn(w)
        return self.out(hn[-1])


class AutoEncoder(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.enc = nn.LSTM(d, h, batch_first=True)
        self.dec = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, d)

    def forward(self, w):                      # восстановление окна (B, WIN, d)
        _, (hn, _) = self.enc(w)
        z = hn[-1].unsqueeze(1).repeat(1, w.shape[1], 1)
        y, _ = self.dec(z)
        return self.out(y)


def windows(x, win=None):
    """Окна x[t-win:t] для t = win..len(x)-1 (прогноз x[t]) — возвращает view."""
    win = win or WIN
    return np.lib.stride_tricks.sliding_window_view(x, (win, x.shape[1]))[:, 0]


def train_model(model, x_tr, kind, epochs, bs=256, lr=1e-3, stride=2):
    w = windows(x_tr)
    if kind == 'fc':
        inp, tgt = w[:-1], x_tr[WIN:]
    else:
        inp = tgt = w
    sel = np.arange(0, len(inp), stride)
    inp_t, tgt_t = torch.from_numpy(np.ascontiguousarray(inp[sel])), torch.from_numpy(np.ascontiguousarray(tgt[sel]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(inp_t), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = ((model(inp_t[b]) - tgt_t[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, kind, bs=2048):
    """Остатки/ошибки для каждой точки x (первые WIN точек — по первому окну)."""
    w = windows(x)
    res = np.zeros_like(x)
    if kind == 'fc':
        for i in range(0, len(w) - 1, bs):
            j = min(i + bs, len(w) - 1)
            res[WIN + i:WIN + j] = x[WIN + i:WIN + j] - model(torch.from_numpy(np.ascontiguousarray(w[i:j]))).numpy()
    else:
        for i in range(0, len(w), bs):
            b = torch.from_numpy(np.ascontiguousarray(w[i:i + bs]))
            res[WIN - 1 + i:WIN - 1 + i + len(b)] = (b - model(b)).numpy()[:, -1]
    return res


def standardize(r, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-3
    return (r - mu) / sd


def standardize_adaptive(r, ref, horizon=POINTS_PER_DAY):
    """Причинная нормировка по экспоненциально взвешенным среднему и дисперсии
    с горизонтом около суток: убирает медленный дрейф остатков."""
    a = 1.0 / horizon
    mu, var = ref.mean(0).astype(np.float64), ref.var(0).astype(np.float64) + 1e-6
    z = np.empty_like(r, dtype=np.float64)
    for t, row in enumerate(r):
        z[t] = (row - mu) / np.sqrt(var)
        d = row - mu
        mu = mu + a * d
        var = (1 - a) * (var + a * d * d)
    return z


# ---------------------------------------------------------------- эксперимент

def run_machine(machine, epochs, rng, hidden=64, methods='all'):
    x_tr, x_te, y = load(machine)
    segs = segments(y)
    n_val = int(len(x_tr) * VAL_FRAC)
    x_fit, x_val = x_tr[:-n_val], x_tr[-n_val:]
    rows, scores = [], {}

    def record(name, s_test, s_val, sec):
        f1, f1_pa = best_f1(s_test, y, segs)
        op = operational(s_test, y, segs, np.quantile(s_val, OP_Q))
        rows.append(dict(machine=machine, method=name, f1=f1, f1_pa=f1_pa, seconds=sec, **op))
        scores[f'{name}_test'] = s_test.astype(np.float32)
        scores[f'{name}_val'] = s_val.astype(np.float32)

    if methods == 'nn':
        return run_nn(machine, x_tr, x_fit, x_val, x_te, epochs, hidden, record) or (rows, scores, y)

    # случайная мера
    t0 = time.perf_counter()
    record('random', rng.random(len(x_te)), rng.random(len(x_val)), time.perf_counter() - t0)

    # CUSUM по остаткам EWMA
    t0 = time.perf_counter()
    r_fit, r_val = ewma_residuals(x_fit), ewma_residuals(np.r_[x_fit[-1:], x_val])[1:]
    r_te = ewma_residuals(np.r_[x_tr[-1:], x_te])[1:]
    t_res = time.perf_counter() - t0
    record('cusum_ewma', cusum(standardize(r_te, r_fit)), cusum(standardize(r_val, r_fit)), time.perf_counter() - t0)
    t0 = time.perf_counter()
    record('cusum_ewma_adapt', cusum(standardize_adaptive(r_te, r_fit)), cusum(standardize_adaptive(r_val, r_fit)),
           t_res + time.perf_counter() - t0)

    # Isolation Forest
    t0 = time.perf_counter()
    iso = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED).fit(x_fit)
    record('iforest', -iso.score_samples(x_te), -iso.score_samples(x_val), time.perf_counter() - t0)

    run_nn(machine, x_tr, x_fit, x_val, x_te, epochs, hidden, record)
    return rows, scores, y


def run_nn(machine, x_tr, x_fit, x_val, x_te, epochs, hidden, record):
    """Нейросетевые методы и гибрид; результаты передаются через record()."""
    # LSTM-прогноз и гибрид LSTM + CUSUM
    torch.manual_seed(SEED)
    t0 = time.perf_counter()
    fc = train_model(Forecaster(x_tr.shape[1], hidden), x_fit, 'fc', epochs)
    r_fit = predict(fc, x_fit, 'fc')[WIN:]
    r_val = predict(fc, np.r_[x_fit[-WIN:], x_val], 'fc')[WIN:]
    r_te = predict(fc, np.r_[x_tr[-WIN:], x_te], 'fc')[WIN:]
    t_fc = time.perf_counter() - t0
    record('lstm', (r_te ** 2).mean(1), (r_val ** 2).mean(1), t_fc)
    t0 = time.perf_counter()
    record('lstm_cusum', cusum(standardize(r_te, r_fit)), cusum(standardize(r_val, r_fit)),
           t_fc + time.perf_counter() - t0)
    t0 = time.perf_counter()
    record('lstm_cusum_adapt', cusum(standardize_adaptive(r_te, r_fit)), cusum(standardize_adaptive(r_val, r_fit)),
           t_fc + time.perf_counter() - t0)

    # LSTM-автоэнкодер
    torch.manual_seed(SEED)
    t0 = time.perf_counter()
    ae = train_model(AutoEncoder(x_tr.shape[1], hidden), x_fit, 'ae', epochs)
    e_val = predict(ae, np.r_[x_fit[-WIN:], x_val], 'ae')[WIN:]
    e_te = predict(ae, np.r_[x_tr[-WIN:], x_te], 'ae')[WIN:]
    record('lstm_ae', (e_te ** 2).mean(1), (e_val ** 2).mean(1), time.perf_counter() - t0)


def main():
    global WIN, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument('--machines', default='')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--win', type=int, default=WIN)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--methods', choices=['all', 'nn'], default='all')
    ap.add_argument('--tag', default='', help='подпапка results/<tag> для прогона')
    ap.add_argument('--save-scores', action='store_true', help='сохранить меры аномальности всех серверов')
    args = ap.parse_args()
    WIN, SEED = args.win, args.seed
    machines = args.machines.split(',') if args.machines else sorted(p.stem for p in (DATA / 'test').glob('*.txt'))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = OUT / args.tag if args.tag else OUT
    out.mkdir(parents=True, exist_ok=True)
    if args.save_scores:
        (out / 'scores').mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_rows = []
    for m in machines:
        t0 = time.perf_counter()
        rows, scores, y = run_machine(m, args.epochs, rng, args.hidden, args.methods)
        all_rows += rows
        pd.DataFrame(all_rows).to_csv(out / 'per_machine.csv', index=False)
        if args.save_scores:
            np.savez_compressed(out / 'scores' / f'{m}.npz', labels=y, **scores)
        print(f'{m}: {time.perf_counter() - t0:.0f} s', flush=True)
    df = pd.DataFrame(all_rows)
    summary = df.groupby('method').agg(
        f1=('f1', 'mean'), f1_pa=('f1_pa', 'mean'), event_recall=('event_recall', 'mean'),
        delay_median=('delay_median', 'median'), false_alarms_per_day=('false_alarms_per_day', 'mean'),
        normal_time_in_alarm=('normal_time_in_alarm', 'mean'), seconds=('seconds', 'mean')).round(3)
    order = ['random', 'cusum_ewma', 'cusum_ewma_adapt', 'iforest', 'lstm', 'lstm_ae', 'lstm_cusum', 'lstm_cusum_adapt']
    summary = summary.reindex([o for o in order if o in summary.index])
    summary.to_csv(out / 'summary.csv')
    (out / 'config.json').write_text(json.dumps(dict(seed=SEED, win=WIN, hidden=args.hidden, val_frac=VAL_FRAC,
                                                     op_q=OP_Q, cusum_k=CUSUM_K, merge_gap=MERGE_GAP,
                                                     epochs=args.epochs, methods=args.methods, machines=machines,
                                                     torch=torch.__version__), ensure_ascii=False, indent=1))
    print(summary.to_string())


if __name__ == '__main__':
    main()
