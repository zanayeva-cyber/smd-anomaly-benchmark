"""Сводка проверки чувствительности.

1) Нейросетевые параметры: results/<tag>/summary.csv для прогонов win15, win60, h32, h128, ep4, ep16, seed1, seed2
   (базовый прогон — results/base).
2) Квантиль эксплуатационного порога: пересчёт по сохранённым мерам results/base/scores/*.npz.
3) Параметр CUSUM k: пересчёт CUSUM по остаткам EWMA для k = 0.25, 0.5, 1.0.
"""
import numpy as np
import pandas as pd

import smd_benchmark as B

R = B.OUT
NN = ['lstm', 'lstm_ae', 'lstm_cusum']
configs = [
    ('base', 'Базовый: окно 30, LSTM 64, 8 эпох, seed 0'),
    ('win15', 'Окно 15 мин'), ('win60', 'Окно 60 мин'),
    ('h32', 'Скрытое состояние 32'), ('h128', 'Скрытое состояние 128'),
    ('ep4', '4 эпохи'), ('ep16', '16 эпох'),
    ('seed1', 'seed 1'), ('seed2', 'seed 2'),
]

# ---- 1. нейросетевые параметры
rows = []
for tag, label in configs:
    p = R / tag / 'summary.csv'
    if not p.exists():
        continue
    s = pd.read_csv(p, index_col='method')
    row = dict(config=tag, label=label)
    for m in NN:
        if m in s.index:
            row[f'{m}_f1'] = s.loc[m, 'f1']
            row[f'{m}_f1_pa'] = s.loc[m, 'f1_pa']
            row[f'{m}_fa'] = s.loc[m, 'false_alarms_per_day']
            row[f'{m}_recall'] = s.loc[m, 'event_recall']
    rows.append(row)
nn_df = pd.DataFrame(rows)
nn_df.to_csv(R / 'sensitivity_nn.csv', index=False)

seeds = nn_df[nn_df.config.isin(['base', 'seed1', 'seed2'])]
seed_stats = {m: (seeds[f'{m}_f1'].mean(), seeds[f'{m}_f1'].std(ddof=1)) for m in NN if f'{m}_f1' in seeds}

# ---- 2. квантиль порога
qs = [0.99, 0.995, 0.999]
q_rows = []
files = sorted((R / 'base' / 'scores').glob('*.npz'))
methods = [k[:-5] for k in np.load(files[0]).files if k.endswith('_test')]
acc = {(m, q): [] for m in methods for q in qs}
for f in files:
    z = np.load(f)
    y = z['labels']
    segs = B.segments(y)
    for m in methods:
        for q in qs:
            acc[(m, q)].append(B.operational(z[f'{m}_test'], y, segs, np.quantile(z[f'{m}_val'], q)))
for (m, q), lst in acc.items():
    d = pd.DataFrame(lst)
    q_rows.append(dict(method=m, q=q, event_recall=d.event_recall.mean(), false_alarms_per_day=d.false_alarms_per_day.mean(),
                       normal_time_in_alarm=d.normal_time_in_alarm.mean()))
q_df = pd.DataFrame(q_rows)
q_df.to_csv(R / 'sensitivity_threshold.csv', index=False)

# ---- 3. параметр k у CUSUM по остаткам EWMA
k_rows = []
machines = [f.stem for f in files]
for k in [0.25, 0.5, 1.0]:
    res = []
    for m in machines:
        x_tr, x_te, y = B.load(m)
        segs = B.segments(y)
        n_val = int(len(x_tr) * B.VAL_FRAC)
        x_fit, x_val = x_tr[:-n_val], x_tr[-n_val:]
        r_fit = B.ewma_residuals(x_fit)
        r_val = B.ewma_residuals(np.r_[x_fit[-1:], x_val])[1:]
        r_te = B.ewma_residuals(np.r_[x_tr[-1:], x_te])[1:]
        s_te, s_val = B.cusum(B.standardize(r_te, r_fit), k), B.cusum(B.standardize(r_val, r_fit), k)
        f1, f1_pa = B.best_f1(s_te, y, segs)
        res.append(dict(f1=f1, f1_pa=f1_pa, **B.operational(s_te, y, segs, np.quantile(s_val, B.OP_Q))))
    d = pd.DataFrame(res)
    k_rows.append(dict(k=k, f1=d.f1.mean(), f1_pa=d.f1_pa.mean(), event_recall=d.event_recall.mean(),
                       false_alarms_per_day=d.false_alarms_per_day.mean(), normal_time_in_alarm=d.normal_time_in_alarm.mean()))
k_df = pd.DataFrame(k_rows)
k_df.to_csv(R / 'sensitivity_cusum_k.csv', index=False)

pd.set_option('display.width', 200)
print('=== NN configs ===')
print(nn_df.round(3).to_string(index=False))
print('=== seeds: mean, std of F1 ===')
for m, (mu, sd) in seed_stats.items():
    print(f'{m}: {mu:.3f} ± {sd:.3f}')
print('=== threshold quantile ===')
print(q_df.round(3).to_string(index=False))
print('=== CUSUM k ===')
print(k_df.round(3).to_string(index=False))
