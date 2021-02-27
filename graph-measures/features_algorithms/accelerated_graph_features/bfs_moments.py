import os
import sys
import numpy as np
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/accelerated_graph_features'))

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from features_algorithms.accelerated_graph_features.src import bfs_moments


class BfsMomentsCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        self._features = bfs_moments(self._gnx)

    def _get_feature(self, element):
        return np.array(self._features[element])


feature_entry = {
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
}
