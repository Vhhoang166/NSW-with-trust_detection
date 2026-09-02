from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from NSW_Moradi_Final import NSWTrustworthySim

from .common import TrustModeStepMixin, normalize_budget


@dataclass
class SampleResolvingParams:
    rho: float = -0.50
    sample_window: int = 30
    current_weight: float = 0.65
    score_power: float = 1.20
    utility_floor: float = 1e-6


class SampleResolvingTrustSim(TrustModeStepMixin, NSWTrustworthySim):
    approach_name = "sample_resolving"

    def __init__(self, *args, sample_params: Optional[SampleResolvingParams] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_params = sample_params or SampleResolvingParams()

    def _sample_stabilized_reports(self, reports: np.ndarray, trusted: List[int]) -> np.ndarray:
        p = self.sample_params
        if not self.hist_reports or not trusted:
            return reports.copy()
        history = np.stack(self.hist_reports[-p.sample_window:], axis=1)
        sample_median = np.median(history, axis=1)
        stable = reports.copy()
        stable[trusted] = (
            p.current_weight * reports[trusted]
            + (1.0 - p.current_weight) * sample_median[trusted]
        )
        return stable

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        alloc = np.zeros(self.N)
        if not trusted:
            alloc[:] = 1.0 / self.N
            return alloc, np.zeros(self.N), alloc.copy(), {"sample_window_used": 0}

        p = self.sample_params
        stable_reports = self._sample_stabilized_reports(reports, trusted)
        baseline = np.maximum(self.utilities[trusted], p.utility_floor)
        marginal = np.maximum(stable_reports[trusted], 0.0) * np.power(
            baseline, p.rho - 1.0
        )
        scores = np.power(np.maximum(marginal, 0.0), p.score_power)
        if scores.sum() <= 1e-12:
            local_alloc = np.ones(len(trusted)) / len(trusted)
        else:
            local_alloc = scores / scores.sum()
        for idx, i in enumerate(trusted):
            alloc[i] = local_alloc[idx]
        alloc = normalize_budget(alloc)
        return alloc, np.zeros(self.N), alloc.copy(), {
            "sample_window_used": min(len(self.hist_reports), p.sample_window),
            "sample_current_weight": p.current_weight,
            "sample_rho": p.rho,
        }
