import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class LoadCentralityCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        self._features = nx.load_centrality(self._gnx)


feature_entry = {
    "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(LoadCentralityCalculator, is_max_connected=True)
