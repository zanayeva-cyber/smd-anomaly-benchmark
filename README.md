# smd-anomaly-benchmark

Код экспериментальной части статьи
**Пашаев М. Я., Занаева З. С. «Методы обнаружения аномалий в телеметрии телекоммуникационного оборудования:
аналитический обзор и сравнительная оценка»** (журнал «Системы управления, связи и безопасности», 2026, направлена в редакцию).

[English version below](#english)

## Что сравнивается

Восемь детекторов аномалий на наборе **Server Machine Dataset (SMD)** — 28 серверов, 38 метрик, интервал 1 мин:

| Ключ | Метод |
|---|---|
| `random` | случайная мера аномальности (контроль протокола оценки) |
| `cusum_ewma` | двусторонний CUSUM по нормированным остаткам EWMA-прогноза |
| `cusum_ewma_adapt` | то же с адаптивной нормировкой (горизонт сутки) |
| `iforest` | Isolation Forest |
| `lstm` | LSTM-прогноз следующего вектора, мера — ошибка прогноза |
| `lstm_ae` | LSTM-автоэнкодер окна, мера — ошибка восстановления |
| `lstm_cusum` | гибрид: CUSUM по остаткам LSTM-прогноза |
| `lstm_cusum_adapt` | гибрид с адаптивной нормировкой |

Метрики: F1 поточечно и F1 с point-adjust (F1-PA) при оптимальном пороге; при эксплуатационном пороге
(99,5-й перцентиль меры на валидационной части обучающей выборки) — доля обнаруженных аномальных интервалов,
медианное запаздывание, число ложных тревог в сутки, доля нормального времени в тревоге.

## Установка и запуск

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_smd.py            # около 460 МБ в data/SMD
python smd_benchmark.py --epochs 8   # около 10 мин на CPU, результаты в results/
python plot_results.py            # рис. 2 статьи
python run_sensitivity.py         # проверка чувствительности, около 1,5 ч
```

Windows: если путь к папке длиннее ~200 символов, PyTorch может не загрузиться (ограничение MAX_PATH) —
создайте виртуальное окружение по короткому пути.

## Основные результаты (табл. 2 статьи, среднее по 28 серверам)

| Метод | F1 | F1-PA | Обнаружено, % | Ложных тревог в сутки | Нормальное время в тревоге, % |
|---|---|---|---|---|---|
| Случайная мера | 0,07 | 0,75 | 27 | 7,3 | 0,5 |
| CUSUM по остаткам EWMA | 0,31 | 0,52 | 60 | 1,2 | 20,0 |
| CUSUM, адаптивная нормировка | 0,30 | 0,48 | 59 | 0,6 | 26,7 |
| Isolation Forest | 0,32 | 0,85 | 66 | 10,2 | 6,9 |
| LSTM-прогноз | 0,43 | 0,96 | 86 | 9,6 | 10,0 |
| LSTM-автоэнкодер | 0,42 | 0,93 | 85 | 9,0 | 11,2 |
| LSTM + CUSUM | 0,20 | 0,25 | 83 | 0,4 | 77,1 |
| LSTM + CUSUM, адаптивная нормировка | 0,15 | 0,24 | 40 | 0,4 | 29,1 |

Проверка чувствительности (окно 15/30/60 мин, LSTM 32/64/128, 4/8/16 эпох, три начальных значения генератора):
F1 LSTM-прогноза 0,41–0,44, LSTM-автоэнкодера 0,40–0,43, гибрида 0,19–0,22; СКО по начальным значениям ≤ 0,007.
Порядок методов сохраняется во всех вариантах. Вычисления на CPU детерминированы: повторный запуск даёт те же числа.

## Состав

- `smd_benchmark.py` — методы, метрики, эксперимент (параметры: `--win`, `--hidden`, `--epochs`, `--seed`,
  `--methods all|nn`, `--tag`, `--save-scores`, `--machines`);
- `analyze_sensitivity.py` — сводка проверки чувствительности, квантиля порога и параметра CUSUM k;
- `plot_results.py` — рисунок F1 / F1-PA;
- `diag_cusum.py` — диагностика тяжёлых хвостов остатков (раздел 7.2 статьи);
- `download_smd.py`, `run_sensitivity.py` — загрузка данных и полный прогон проверки чувствительности;
- `results/` — полученные результаты: `summary.csv` (табл. 2), `per_machine.csv`, `sensitivity_nn.csv` (табл. 3),
  `sensitivity_threshold.csv`, `sensitivity_cusum_k.csv`, `fig2_f1_vs_pa.png`, подпапки прогонов.

## Замечания

- Оригинальный OmniAnomaly требует TensorFlow 1.x, поэтому класс реконструкционных моделей представлен
  LSTM-автоэнкодером; это оговорено в статье.
- Набор SMD в репозиторий не входит: он загружается из репозитория авторов
  ([NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)) и распространяется на их условиях.

## Лицензия и цитирование

Код — под лицензией MIT (`LICENSE`). При использовании просьба ссылаться на статью (`CITATION.cff`).

---

<a name="english"></a>
## English

Code for the experimental part of the paper **Pashaev M. Ya., Zanaeva Z. S. "Anomaly detection methods for
telecommunication equipment telemetry: an analytical review and comparative assessment"** (Systems of Control,
Communication and Security, 2026, submitted).

Eight anomaly detectors are compared on the **Server Machine Dataset (SMD)** (28 servers, 38 metrics, 1-min
sampling): a random score (protocol control), two-sided CUSUM on standardized EWMA-forecast residuals (static and
adaptive normalization), Isolation Forest, an LSTM forecaster, an LSTM autoencoder, and a hybrid CUSUM on LSTM
residuals (static and adaptive normalization). Metrics: point-wise F1 and point-adjusted F1 (F1-PA) at the best
threshold; at an operational threshold (99.5th percentile of the score on a validation split) — share of detected
anomaly segments, median detection delay, false alarms per day and share of normal time spent in alarm.

Quick start:

```
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_smd.py
python smd_benchmark.py --epochs 8
python plot_results.py
python run_sensitivity.py
```

Key findings: point-adjust inflates F1 by an order of magnitude (random score: 0.07 → 0.75); the best neural models
reach point-wise F1 ≈ 0.43 but raise 9–10 false alarms per day per server at an operational threshold; CUSUM-based
detectors and the naive hybrid suffer from heavy-tailed and drifting residuals. Results are robust to window length,
model size, number of epochs and random seed (F1 spread ≤ 0.03). OmniAnomaly is replaced by an LSTM autoencoder
because the original requires TensorFlow 1.x. The SMD dataset is not redistributed; it is downloaded from
[NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly). Code is released under the MIT License.
