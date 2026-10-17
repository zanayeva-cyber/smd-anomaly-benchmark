"""Диагностика: какие метрики держат статистику CUSUM высокой вне аномалий."""
import numpy as np
import smd_benchmark as B

for m in ['machine-1-1', 'machine-2-1', 'machine-3-1']:
    x_tr, x_te, y = B.load(m)
    n_val = int(len(x_tr) * B.VAL_FRAC)
    x_fit = x_tr[:-n_val]
    r_fit = B.ewma_residuals(x_fit)
    r_te = B.ewma_residuals(np.r_[x_tr[-1:], x_te])[1:]
    sd = r_fit.std(0)
    z = B.standardize(r_te, r_fit)
    normal = y == 0
    big = (np.abs(z[normal]) > 4).mean(0)
    const_tr = np.flatnonzero(x_fit.std(0) < 1e-6)
    print(m, 'constant-in-train dims:', const_tr.tolist())
    top = np.argsort(-big)[:6]
    print('  dims with |z|>4 most often on normal test points:',
          [(int(i), round(float(big[i]), 3), float(f'{sd[i]:.2e}')) for i in top])
