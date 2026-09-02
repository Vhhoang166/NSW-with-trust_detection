from .adaptive_alpha import AdaptiveAlphaTrustSim
from .baseline_fixed_alpha import (
    BaselineFixedAlphaSim,
    PureEqualAlpha1Sim,
    PureGreedyAlpha0Sim,
)
from .expert_advice import ExpertAdviceTrustSim
from .generalized_mean import GeneralizedMeanTrustSim
from .pace_trusted import PaceTrustSim
from .robust_aggregation import RobustAggregationTrustSim
from .sample_resolving import SampleResolvingTrustSim


APPROACH_CLASSES = {
    "baseline_fixed_alpha": BaselineFixedAlphaSim,
    "adaptive_alpha": AdaptiveAlphaTrustSim,
    "pace_trusted": PaceTrustSim,
    "generalized_mean_greedy": GeneralizedMeanTrustSim,
    "sample_resolving": SampleResolvingTrustSim,
    "expert_advice": ExpertAdviceTrustSim,
    "robust_aggregation": RobustAggregationTrustSim,
    "pure_greedy_alpha0": PureGreedyAlpha0Sim,
    "pure_equal_alpha1": PureEqualAlpha1Sim,
}
