"""Проверка чувствительности (табл. 3 статьи): базовый прогон и восемь вариантов параметров
нейросетевых моделей, затем сводный анализ. Около 1,5 ч на CPU.

Sensitivity analysis (Table 3 of the paper): a baseline run and eight variations of the neural
model parameters, followed by the summary analysis. About 1.5 h on a CPU.

    python run_sensitivity.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = [
    ['--epochs', '8', '--tag', 'base', '--save-scores'],
    ['--epochs', '8', '--methods', 'nn', '--win', '15', '--tag', 'win15'],
    ['--epochs', '8', '--methods', 'nn', '--win', '60', '--tag', 'win60'],
    ['--epochs', '8', '--methods', 'nn', '--hidden', '32', '--tag', 'h32'],
    ['--epochs', '8', '--methods', 'nn', '--hidden', '128', '--tag', 'h128'],
    ['--epochs', '4', '--methods', 'nn', '--tag', 'ep4'],
    ['--epochs', '16', '--methods', 'nn', '--tag', 'ep16'],
    ['--epochs', '8', '--methods', 'nn', '--seed', '1', '--tag', 'seed1'],
    ['--epochs', '8', '--methods', 'nn', '--seed', '2', '--tag', 'seed2'],
]


def main():
    for args in RUNS:
        print('===', ' '.join(args), flush=True)
        subprocess.run([sys.executable, 'smd_benchmark.py', *args], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'analyze_sensitivity.py'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
