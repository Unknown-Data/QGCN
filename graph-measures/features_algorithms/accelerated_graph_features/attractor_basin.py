import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/accelerated_graph_features'))

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from features_algorithms.accelerated_graph_features.src import attraction_basin


class AttractorBasinCalculator(NodeFeatureCalculator):
    def __init__(self, *args, alpha=2, **kwargs):
        super(AttractorBasinCalculator, self).__init__(*args, **kwargs)
        self._alpha = alpha
        self._default_val = float('nan')

    def is_relevant(self):
        return self._gnx.is_directed()

    def _calculate(self, include: set):
        self._features = attraction_basin(self._gnx, alpha=self._alpha)


feature_entry = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
}
