#!/usr/bin/env python3
"""Compare the existing SLSQP and Frank-Wolfe offline NSW solvers at T=500.

Both methods receive exactly the same deterministic valuation matrix.  The
SLSQP formulation intentionally mirrors ``NSW_Moradi_Final.offline_optimal_nsw``
so this script measures the implementation currently used by the project.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

import NSW_Moradi_Final as moradi
from NSW_Moradi_FrankWolfe import frank_wolfe_offline_nsw


N_LEGITIMATE = 10
N_MALICIOUS = 4
N_ROUNDS = 500
SEED = 42


def slsqp_offline_nsw(
    v_true: np.ndarray,
    legit_mask: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, object]]:
    """Run the same SLSQP formulation used by the original Moradi program."""
    started = time.perf_counter()
    idx = np.where(np.asarray(legit_mask, dtype=bool))[0]
    values = np.asarray(v_true[idx, :], dtype=float)
    n_agents, n_rounds = values.shape

    def objective(flat_allocation: np.ndarray) -> float:
        allocation = flat_allocation.reshape(n_agents, n_rounds)
        utilities = (values * allocation).sum(axis=1)
        if np.any(utilities <= 0):
            return 1e6
        return float(-np.sum(np.log(utilities)))

    constraints = [
        {
            "type": "eq",
            "fun": lambda flat_allocation, t=t: (
                np.sum(flat_allocation.reshape(n_agents, n_rounds)[:, t]) - 1.0
            ),
        }
        for t in range(n_rounds)
    ]
    initial = np.full(n_agents * n_rounds, 1.0 / n_agents)
    bounds = [(0.0, 1.0)] * (n_agents * n_rounds)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    # Match the original program's fallback exactly when SLSQP reports failure.
    if result.success:
        allocation = result.x.reshape(n_agents, n_rounds)
    else:
        allocation = np.full((n_agents, n_rounds), 1.0 / n_agents)
    utilities = (values * allocation).sum(axis=1)
    elapsed = time.perf_counter() - started
    diagnostics: Dict[str, object] = {
        "elapsed_seconds": elapsed,
        "success": bool(result.success),
        "iterations": int(result.nit),
        "message": str(result.message),
    }
    return moradi.compute_nsw(utilities), utilities, allocation, diagnostics


def maximum_budget_error(allocation: np.ndarray) -> float:
    return float(np.max(np.abs(allocation.sum(axis=0) - 1.0)))


def main() -> None:
    rng = np.random.RandomState(SEED)
    n_agents = N_LEGITIMATE + N_MALICIOUS
    valuations = moradi._generate_valuations(
        rng,
        n_agents,
        N_ROUNDS,
        "lognormal",
    )
    legitimate_mask = np.zeros(n_agents, dtype=bool)
    legitimate_mask[:N_LEGITIMATE] = True

    print("Offline NSW solver comparison")
    print(
        f"Dataset: {N_LEGITIMATE} legitimate + {N_MALICIOUS} malicious agents, "
        f"{N_ROUNDS} rounds, seed={SEED}"
    )
    print("Running SLSQP ...")
    slsqp_nsw, _, slsqp_allocation, slsqp_info = slsqp_offline_nsw(
        valuations,
        legitimate_mask,
    )

    print("Running Frank-Wolfe ...")
    fw_nsw, _, fw_allocation, fw_info = frank_wolfe_offline_nsw(
        valuations,
        legitimate_mask,
        report=False,
    )

    absolute_difference = abs(fw_nsw - slsqp_nsw)
    relative_difference = (
        absolute_difference / abs(slsqp_nsw) if slsqp_nsw != 0 else float("nan")
    )
    speedup = (
        float(slsqp_info["elapsed_seconds"]) / float(fw_info["elapsed_seconds"])
        if float(fw_info["elapsed_seconds"]) > 0
        else float("inf")
    )

    print("\nResults")
    print(
        f"SLSQP       NSW={slsqp_nsw:.10f}  "
        f"time={float(slsqp_info['elapsed_seconds']):.3f}s  "
        f"iterations={slsqp_info['iterations']}  success={slsqp_info['success']}"
    )
    print(f"             message={slsqp_info['message']}")
    print(
        f"Frank-Wolfe NSW={fw_nsw:.10f}  "
        f"time={float(fw_info['elapsed_seconds']):.3f}s  "
        f"iterations={int(fw_info['iterations'])}  "
        f"gap={float(fw_info['final_gap']):.3e}  "
        f"converged={bool(fw_info['converged'])}"
    )
    print(f"Absolute NSW difference: {absolute_difference:.10e}")
    print(f"Relative NSW difference: {relative_difference:.6%}")
    print(f"Measured speedup:         {speedup:.1f}x")
    print(f"SLSQP max budget error:   {maximum_budget_error(slsqp_allocation):.3e}")
    print(f"FW max budget error:      {maximum_budget_error(fw_allocation):.3e}")


if __name__ == "__main__":
    main()
