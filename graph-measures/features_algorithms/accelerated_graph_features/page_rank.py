import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/accelerated_graph_features'))

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from features_algorithms.accelerated_graph_features.src import node_page_rank


class PageRankCalculator(NodeFeatureCalculator):
    def __init__(self, *args, alpha=0.9, **kwargs):
        super(PageRankCalculator, self).__init__(*args, **kwargs)
        self._alpha = alpha

    def is_relevant(self):
        # Undirected graphs will be converted to a directed
        #       graph with two directed edges for each undirected edge.
        return True

    def _calculate(self, include: set):
        self._features = node_page_rank(self._gnx, dumping=self._alpha)


feature_entry = {
    "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
}
