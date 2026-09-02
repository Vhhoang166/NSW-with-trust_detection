from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from NSW_Moradi_Final import NSWTrustworthySim

from .common import TrustModeStepMixin, normalize_budget


@dataclass
class GeneralizedMeanParams:
    rho: float = -0.50
    score_power: float = 1.25
    utility_floor: float = 1e-6
    prediction_floor_weight: float = 0.25


class GeneralizedMeanTrustSim(TrustModeStepMixin, NSWTrustworthySim):
    approach_name = "generalized_mean_greedy"

    def __init__(self, *args, gm_params: Optional[GeneralizedMeanParams] = None, **kwargs):
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
        marginal = values * np.power(baseline, p.rho - 1.0)
        scores = np.power(np.maximum(marginal, 0.0), p.score_power)
        if scores.sum() <= 1e-12:
            local_alloc = np.ones(len(trusted)) / len(trusted)
        else:
            local_alloc = scores / scores.sum()
        for idx, i in enumerate(trusted):
            alloc[i] = local_alloc[idx]
        alloc = normalize_budget(alloc)
        return alloc, np.zeros(self.N), alloc.copy(), {
            "gm_rho": p.rho,
            "gm_avg_baseline": float(np.mean(baseline)),
            "gm_min_baseline": float(np.min(baseline)),
        }
