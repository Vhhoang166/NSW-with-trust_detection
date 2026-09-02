from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from NSW_Moradi_Final import NSWTrustworthySim

from .common import TrustModeStepMixin, normalize_budget


@dataclass
class PaceParams:
    eta: float = 0.08
    price_min: float = 0.20
    price_max: float = 5.00
    score_power: float = 1.50
    utility_floor: float = 1e-6


class PaceTrustSim(TrustModeStepMixin, NSWTrustworthySim):
    approach_name = "pace_trusted"

    def __init__(self, *args, pace_params: Optional[PaceParams] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pace_params = pace_params or PaceParams()
        self.pacing_prices = np.ones(self.N)

    def _update_prices(self, trusted: List[int]) -> None:
        if not trusted:
            return
        p = self.pace_params
        target = float(np.mean(self.utilities[trusted]) + p.utility_floor)
        for i in trusted:
            ratio = (self.utilities[i] + p.utility_floor) / target
            self.pacing_prices[i] *= np.exp(p.eta * np.clip(ratio - 1.0, -2.0, 2.0))
            self.pacing_prices[i] = float(
                np.clip(self.pacing_prices[i], p.price_min, p.price_max)
            )

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        alloc = np.zeros(self.N)
        if not trusted:
            alloc[:] = 1.0 / self.N
            return alloc, np.zeros(self.N), alloc.copy(), {"pace_avg_price": 1.0}

        self._update_prices(trusted)
        p = self.pace_params
        values = np.maximum(reports[trusted], 0.0)
        bang_per_price = values / np.maximum(self.pacing_prices[trusted], p.price_min)
        scores = np.power(np.maximum(bang_per_price, 0.0), p.score_power)
        if scores.sum() <= 1e-12:
            local_alloc = np.ones(len(trusted)) / len(trusted)
        else:
            local_alloc = scores / scores.sum()
        for idx, i in enumerate(trusted):
            alloc[i] = local_alloc[idx]
        alloc = normalize_budget(alloc)
        return alloc, np.zeros(self.N), alloc.copy(), {
            "pace_avg_price": float(np.mean(self.pacing_prices[trusted])),
            "pace_min_price": float(np.min(self.pacing_prices[trusted])),
            "pace_max_price": float(np.max(self.pacing_prices[trusted])),
        }
