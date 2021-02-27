import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/accelerated_graph_features'))

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from features_algorithms.accelerated_graph_features.src import flow


class FlowCalculator(NodeFeatureCalculator):
    """See Y. Rozen & Y. Louzoun article <add-link>"""

    def __init__(self, *args, threshold=0, **kwargs):
        super(FlowCalculator, self).__init__(*args, **kwargs)
        self._threshold = threshold

    def is_relevant(self):
        return self._gnx.is_directed()

    def _calculate(self, include):
        self._features = flow(self._gnx, threshold=self._threshold)


feature_entry = {
    "flow": FeatureMeta(FlowCalculator, {}),
}
