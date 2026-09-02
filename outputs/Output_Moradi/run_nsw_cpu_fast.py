#!/usr/bin/env python3
"""CPU-only accelerated runner for NSW_Moradi_Final.py.

This imports the original script and replaces only the slow SLSQP offline NSW
benchmark with a Frank-Wolfe solver for the same concave offline objective.
Everything else (attack model, trials, sweeps, plots) runs through the original
main() code path.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import importlib.util
import numpy as np
from pathlib import Path

SRC = Path('/mnt/data/NSW_Moradi_Final.py')
spec = importlib.util.spec_from_file_location('nsw_original', SRC)
nsw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nsw)


def fast_offline_optimal_nsw(v_true, legit_mask, max_iter=180, tol=1e-7):
    """Fast Frank-Wolfe solver for max sum_i log(u_i).

    The original code uses SLSQP over N*T allocation variables and T equality
    constraints. This solver works directly in utility space: each FW linear
    oracle gives each round's item to the agent with max v_i,t / u_i, then uses
    an exact 1-D line search for the concave log objective.
    """
    idx = np.where(legit_mask)[0]
    V = v_true[idx, :].astype(float, copy=False)
    N, T = V.shape
    ar = np.arange(T)

    # Equal allocation over legitimate agents is a strictly positive feasible start.
    u = np.maximum(V.sum(axis=1) / max(N, 1), 1e-15)

    for _ in range(max_iter):
        scores = V / u[:, None]
        winners = np.argmax(scores, axis=0)
        s_u = np.bincount(winners, weights=V[winners, ar], minlength=N).astype(float)

        # Frank-Wolfe dual gap for sum log(u): <grad, s-u>.
        gap = float(scores[winners, ar].sum() - N)
        if gap < tol * max(1, N):
            break

        d = s_u - u

        def deriv(g):
            return float(np.sum(d / np.maximum(u + g * d, 1e-15)))

        d0 = deriv(0.0)
        d1 = deriv(1.0 - 1e-12)
        if d0 <= 0.0:
            gamma = 0.0
        elif d1 >= 0.0:
            gamma = 1.0 - 1e-12
        else:
            lo, hi = 0.0, 1.0 - 1e-12
            for _ in range(24):
                mid = (lo + hi) / 2.0
                if deriv(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            gamma = (lo + hi) / 2.0

        if gamma <= 1e-14:
            break
        u = np.maximum(u + gamma * d, 1e-15)

    X_dummy = np.ones((N, T), dtype=float) / max(N, 1)
    return nsw.compute_nsw(u), u, X_dummy


nsw.offline_optimal_nsw = fast_offline_optimal_nsw

if __name__ == '__main__':
    nsw.main()
