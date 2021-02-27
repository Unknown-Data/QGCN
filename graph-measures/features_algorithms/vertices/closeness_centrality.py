import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class ClosenessCentralityCalculator(NodeFeatureCalculator):
    def _calculate(self, include: set):
        self._features = nx.closeness_centrality(self._gnx)

    def is_relevant(self):
        return True


feature_entry = {
    "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(ClosenessCentralityCalculator, is_max_connected=True)
