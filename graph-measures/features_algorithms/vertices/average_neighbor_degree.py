import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class AverageNeighborDegreeCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        self._features = nx.average_neighbor_degree(self._gnx)


feature_entry = {
    "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(AverageNeighborDegreeCalculator, is_max_connected=True)
