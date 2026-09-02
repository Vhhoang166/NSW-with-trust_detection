"""
Approach 3: Generalized-mean greedy allocator on the trusted set.

This variant replaces the Nash-specific set-aside greedy split with a one-step
generalized-mean marginal rule.  For rho -> 0, the marginal priority resembles
Nash welfare (v_i / u_i).  For rho < 0, low-utility agents receive stronger
priority; for rho > 0, the rule becomes less aggressively fairness-seeking.

The implementation is deliberately lightweight so it can plug into the current
simulator without adding another expensive optimizer inside every round.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from approach_common import VariantStepMixin, normalize_budget
from NSW_Moradi_Final import NSWTrustworthySim


@dataclass
class GeneralizedMeanParams:
    rho: float = -0.50
    score_power: float = 1.25
    utility_floor: float = 1e-6
    prediction_floor_weight: float = 0.25


class GeneralizedMeanGreedySim(VariantStepMixin, NSWTrustworthySim):
    approach_name = "generalized_mean_greedy"

    def __init__(self, *args, gm_params: GeneralizedMeanParams = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gm_params = gm_params or GeneralizedMeanParams()

    def _baseline_utility(self, trusted: List[int]) -> np.ndarray:
        p = self.gm_params
        pred_floor = p.prediction_floor_weight * self.V_tilde[trusted] / max(self.T, 1)
        return np.maximum(self.utilities[trusted] + pred_floor, p.utility_floor)

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        alloc = np.zeros(self.N)
        if not trusted:
            alloc[:] = 1.0 / self.N
            return alloc, np.zeros(self.N), alloc.copy(), {"gm_rho": self.gm_params.rho}

        p = self.gm_params
        values = np.maximum(reports[trusted], 0.0)
        baseline = self._baseline_utility(trusted)

        # Marginal derivative of generalized mean utility is proportional to
        # u_i^(rho-1).  Multiplying by v_i gives the one-round marginal value.
        marginal = values * np.power(baseline, p.rho - 1.0)
        scores = np.power(np.maximum(marginal, 0.0), p.score_power)
        if scores.sum() <= 1e-12:
            local_alloc = np.ones(len(trusted)) / len(trusted)
        else:
            local_alloc = scores / scores.sum()

        for idx, i in enumerate(trusted):
            alloc[i] = local_alloc[idx]
        alloc = normalize_budget(alloc)

        meta = {
            "gm_rho": p.rho,
            "gm_avg_baseline": float(np.mean(baseline)),
            "gm_min_baseline": float(np.min(baseline)),
        }
        return alloc, np.zeros(self.N), alloc.copy(), meta


def run_generalized_mean_trial(config, seed: int = 42):
    sim = GeneralizedMeanGreedySim(config, seed=seed)
    return sim.run(verbose=False), sim

