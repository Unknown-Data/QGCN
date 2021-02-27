import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/accelerated_graph_features'))

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from features_algorithms.accelerated_graph_features.src import k_core


class KCoreCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        self._features = k_core(self._gnx)


feature_entry = {
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
}
