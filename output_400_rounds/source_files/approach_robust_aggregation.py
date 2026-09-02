"""
Approach 6: Robust aggregation before allocation.

This is a supporting robustness layer rather than a wholly new welfare rule.
It keeps the trusted set-aside greedy allocator but clips trusted reports using
a median/MAD envelope before allocation.  The goal is to reduce the damage from
surviving attackers whose reports are extreme relative to other trusted agents.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from approach_common import VariantStepMixin
from NSW_Moradi_Final import NSWTrustworthySim, trusted_online_allocation


@dataclass
class RobustAggregationParams:
    mad_multiplier: float = 3.0
    min_width: float = 1e-4
    alpha: float = None


class RobustAggregationSim(VariantStepMixin, NSWTrustworthySim):
    approach_name = "robust_aggregation"

    def __init__(self, *args, robust_params: RobustAggregationParams = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.robust_params = robust_params or RobustAggregationParams()

    def _clip_reports(self, reports: np.ndarray, trusted: List[int]) -> np.ndarray:
        clipped = reports.copy()
        if len(trusted) < 3:
            return clipped
        vals = np.maximum(reports[trusted], 0.0)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        width = max(self.robust_params.mad_multiplier * 1.4826 * mad,
                    self.robust_params.min_width)
        lo = max(0.0, med - width)
        hi = med + width
        clipped[trusted] = np.clip(vals, lo, hi)
        return clipped

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        robust_reports = self._clip_reports(reports, trusted)
        alpha = (
            self.cfg.allocation_alpha
            if self.robust_params.alpha is None
            else self.robust_params.alpha
        )
        alloc, set_aside, greedy = trusted_online_allocation(
            reports=robust_reports,
            trusted_ids=trusted,
            V_tilde=self.V_tilde,
            u_greedy=self.u_greedy,
            budget=1.0,
            alpha=alpha,
        )
        meta = {
            "robust_alpha": float(alpha),
            "robust_report_delta": float(np.mean(np.abs(robust_reports - reports))),
        }
        return alloc, set_aside, greedy, meta


def run_robust_aggregation_trial(config, seed: int = 42):
    sim = RobustAggregationSim(config, seed=seed)
    return sim.run(verbose=False), sim

