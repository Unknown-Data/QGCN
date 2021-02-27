import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class EccentricityCalculator(NodeFeatureCalculator):
    def _calculate(self, include: set):
        dists = {src: neighbors for src, neighbors in nx.all_pairs_shortest_path_length(self._gnx)}
        self._features = {node: max(neighbors.values()) for node, neighbors in dists.items()}

    def _calculate_dep(self, include: set):
        # Not using eccentricity to handle disconnected graphs. (If a graph has more than 1 connected components,
        # the eccentricty will raise an exception)
        self._features = {node: nx.eccentricity(self._gnx, node) for node in self._gnx}

    def is_relevant(self):
        return True


feature_entry = {
    "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(EccentricityCalculator, is_max_connected=True)
