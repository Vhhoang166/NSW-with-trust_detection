from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from NSW_Moradi_Final import NSWTrustworthySim, trusted_online_allocation

from .common import TrustModeStepMixin


@dataclass
class AdaptiveAlphaParams:
    alpha_min: float = 0.20
    alpha_max: float = 0.85
    trust_weight: float = 0.65
    prediction_weight: float = 0.35
    smoothing: float = 0.80
    mismatch_scale: float = 2.0


class AdaptiveAlphaTrustSim(TrustModeStepMixin, NSWTrustworthySim):
    approach_name = "adaptive_alpha"

    def __init__(self, *args, adaptive_params: Optional[AdaptiveAlphaParams] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_params = adaptive_params or AdaptiveAlphaParams()
        self.alpha_t = float(self.cfg.allocation_alpha)

    def _trust_instability(self) -> float:
        if not self.legit_ids:
            return 0.0
        votes = np.zeros(self.N)
        for i in self.legit_ids:
            _, detected = self.detectors[i].classify(self.t)
            for j in detected:
                votes[j] += 1.0
        p = votes / max(len(self.legit_ids), 1)
        disagreement = 4.0 * p * (1.0 - p)
        return float(np.clip(np.mean(disagreement), 0.0, 1.0))

    def _prediction_instability(self, reports: np.ndarray, trusted: List[int]) -> float:
        if not trusted:
            return 1.0
        predicted_round = self.V_tilde / max(self.T, 1)
        ratios = np.abs(reports[trusted] - predicted_round[trusted]) / (
            predicted_round[trusted] + 1e-10
        )
        mismatch = float(np.median(ratios))
        return float(np.clip(mismatch / self.adaptive_params.mismatch_scale, 0.0, 1.0))

    def _choose_alpha(self, reports: np.ndarray, trusted: List[int]) -> Tuple[float, Dict]:
        p = self.adaptive_params
        trust_risk = self._trust_instability()
        prediction_risk = self._prediction_instability(reports, trusted)
        risk = float(np.clip(
            p.trust_weight * trust_risk + p.prediction_weight * prediction_risk,
            0.0,
            1.0,
        ))
        raw_alpha = p.alpha_min + (p.alpha_max - p.alpha_min) * risk
        self.alpha_t = p.smoothing * self.alpha_t + (1.0 - p.smoothing) * raw_alpha
        self.alpha_t = float(np.clip(self.alpha_t, p.alpha_min, p.alpha_max))
        return self.alpha_t, {
            "alpha_t": self.alpha_t,
            "trust_risk": trust_risk,
            "prediction_risk": prediction_risk,
            "controller_risk": risk,
        }

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        alpha_t, meta = self._choose_alpha(reports, trusted)
        alloc, set_aside, greedy = trusted_online_allocation(
            reports=reports,
            trusted_ids=trusted,
            V_tilde=self.V_tilde,
            u_greedy=self.u_greedy,
            budget=1.0,
            alpha=alpha_t,
        )
        return alloc, set_aside, greedy, meta
