import networkx as nx
import networkx.linalg.algebraicconnectivity as alg_connectivity

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class FiedlerVectorCalculator(NodeFeatureCalculator):
    def _calculate_dep(self, include: set):
        # Working on every connected component by itself
        self._features = dict(zip(self._gnx, alg_connectivity.fiedler_vector(self._gnx)))

    def _calculate(self, include: set):
        self._features = {}

        for graph in nx.connected_component_subgraphs(self._gnx):
            if len(graph) < 2:
                self._features.update(zip(graph.nodes(), [0.] * len(graph)))
            else:
                self._features.update(zip(graph.nodes(), map(float, alg_connectivity.fiedler_vector(graph))))

    def is_relevant(self):
        # Fiedler vector also works only on connected undirected graphs
        # so if gnx is not connected we shall expect an exception: networkx.exception.NetworkXError
        # return (not self._gnx.is_directed()) and (nx.is_connected(self._gnx.to_undirected()))
        return not self._gnx.is_directed()


feature_entry = {
    "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(FiedlerVectorCalculator, is_max_connected=True)
