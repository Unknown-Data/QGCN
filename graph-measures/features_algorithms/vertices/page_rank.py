import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class PageRankCalculator(NodeFeatureCalculator):
    def __init__(self, *args, alpha=0.9, **kwargs):
        super(PageRankCalculator, self).__init__(*args, **kwargs)
        self._alpha = alpha

    def is_relevant(self):
        # Undirected graphs will be converted to a directed
        #       graph with two directed edges for each undirected edge.
        return True

    def _calculate(self, include: set):
        self._features = nx.pagerank(self._gnx, alpha=self._alpha)


feature_entry = {
    "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
}

if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(PageRankCalculator, is_max_connected=True)
