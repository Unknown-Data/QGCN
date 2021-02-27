import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class CommunicabilityBetweennessCentralityCalculator(NodeFeatureCalculator):
    def _calculate(self, include: set):
        self._features = nx.communicability_betweenness_centrality(self._gnx)

    def is_relevant(self):
        return not self._gnx.is_directed()


feature_entry = {
    "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
                                                          {"communicability"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(CommunicabilityBetweennessCentralityCalculator, is_max_connected=True)
