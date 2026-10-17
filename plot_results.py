"""Рис. 2 для статьи: F1 поточечно и F1 с point-adjust по методам (среднее по 28 серверам SMD)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
s = pd.read_csv(ROOT / 'results' / 'summary.csv', index_col='method')
names = {
    'random': 'Случайная мера', 'cusum_ewma': 'CUSUM (EWMA)', 'cusum_ewma_adapt': 'CUSUM (EWMA, адапт.)',
    'iforest': 'Isolation Forest', 'lstm': 'LSTM-прогноз', 'lstm_ae': 'LSTM-автоэнкодер',
    'lstm_cusum': 'LSTM + CUSUM', 'lstm_cusum_adapt': 'LSTM + CUSUM (адапт.)',
}
s = s.loc[[m for m in names if m in s.index]][::-1]

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11})
fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
y = range(len(s))
h = 0.36
# цвета — слоты 1 и 2 проверенной категориальной палитры (синий, оранжевый); текст — нейтральный
ink, ink2, grid = '#0b0b0b', '#52514e', '#e1e0d9'
b1 = ax.barh([i + h / 2 + 0.02 for i in y], s['f1_pa'], height=h, color='#eb6834', label='F1 с point-adjust')
b2 = ax.barh([i - h / 2 - 0.02 for i in y], s['f1'], height=h, color='#2a78d6', label='F1 поточечно')
for bars, col in ((b1, 'f1_pa'), (b2, 'f1')):
    for bar, v in zip(bars, s[col]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f'{v:.2f}'.replace('.', ','), va='center', fontsize=9, color=ink)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))
ax.set_yticks(list(y))
ax.set_yticklabels([names[m] for m in s.index], color=ink)
ax.set_xlim(0, 1.08)
ax.set_xlabel('F1 (среднее по 28 серверам SMD)', color=ink2)
ax.xaxis.grid(True, color=grid, linewidth=0.6)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.spines['left'].set_color(ink2); ax.spines['bottom'].set_color(ink2)
ax.tick_params(colors=ink2)
ax.legend(loc='lower right', frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / 'results' / 'fig2_f1_vs_pa.png')
print('saved')
