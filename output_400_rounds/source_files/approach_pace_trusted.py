"""
Approach 2: PACE-inspired allocator on the trusted set.

This file is intentionally a benchmark, not the recommended replacement.  It
removes the set-aside floor and uses pacing prices to push allocation toward
trusted agents that are currently under-served.  Because attackers that survive
the trust filter can still distort reports, the implementation clips prices and
uses smooth proportional allocation instead of brittle winner-take-all pacing.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from approach_common import VariantStepMixin, normalize_budget
from NSW_Moradi_Final import NSWTrustworthySim


@dataclass
class PaceParams:
    eta: float = 0.08
    price_min: float = 0.20
    price_max: float = 5.00
    score_power: float = 1.50
    utility_floor: float = 1e-6


class PaceTrustedSim(VariantStepMixin, NSWTrustworthySim):
    approach_name = "pace_trusted"

    def __init__(self, *args, pace_params: PaceParams = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pace_params = pace_params or PaceParams()
        self.pacing_prices = np.ones(self.N)

    def _update_prices(self, trusted: List[int]) -> None:
        if not trusted:
            return
        p = self.pace_params
        util = self.utilities[trusted]
        target = float(np.mean(util) + p.utility_floor)
        for i in trusted:
            ratio = (self.utilities[i] + p.utility_floor) / target
            # Over-served agents get a higher price; under-served agents get
            # a lower price.  Clipping prevents a poisoned history from making
            # future allocations numerically degenerate.
            self.pacing_prices[i] *= np.exp(p.eta * np.clip(ratio - 1.0, -2.0, 2.0))
            self.pacing_prices[i] = float(
                np.clip(self.pacing_prices[i], p.price_min, p.price_max))

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

        meta = {
            "pace_avg_price": float(np.mean(self.pacing_prices[trusted])),
            "pace_min_price": float(np.min(self.pacing_prices[trusted])),
            "pace_max_price": float(np.max(self.pacing_prices[trusted])),
        }
        # The whole allocation is treated as the greedy/adaptive component.
        return alloc, np.zeros(self.N), alloc.copy(), meta


def run_pace_trial(config, seed: int = 42):
    sim = PaceTrustedSim(config, seed=seed)
    return sim.run(verbose=False), sim

