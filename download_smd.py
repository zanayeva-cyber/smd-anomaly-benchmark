"""Загрузка набора Server Machine Dataset (SMD) из репозитория авторов OmniAnomaly.

Download the Server Machine Dataset (SMD) from the OmniAnomaly authors' repository
(https://github.com/NetManAIOps/OmniAnomaly, folder ServerMachineDataset) into data/SMD.

    python download_smd.py
"""
import urllib.request
from pathlib import Path

BASE = 'https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset'
OUT = Path(__file__).resolve().parent / 'data' / 'SMD'
MACHINES = ([f'machine-1-{i}' for i in range(1, 9)] + [f'machine-2-{i}' for i in range(1, 10)]
            + [f'machine-3-{i}' for i in range(1, 12)])


def main():
    for sub in ('train', 'test', 'test_label'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for m in MACHINES:
            dst = OUT / sub / f'{m}.txt'
            if dst.exists():
                continue
            print(f'{sub}/{m}.txt', flush=True)
            urllib.request.urlretrieve(f'{BASE}/{sub}/{m}.txt', dst)
    print(f'done: {OUT}')


if __name__ == '__main__':
    main()
