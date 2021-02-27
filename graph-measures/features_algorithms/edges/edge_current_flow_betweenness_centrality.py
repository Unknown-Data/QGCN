import networkx as nx

from features_infra.feature_calculators import EdgeFeatureCalculator, FeatureMeta


class EdgeCurrentFlowCalculator(EdgeFeatureCalculator):
    def _calculate(self, include: set):
        self._features = nx.edge_current_flow_betweenness_centrality(self._gnx)

    def is_relevant(self):
        return True


feature_entry = {
    "edge_current_flow": FeatureMeta(EdgeCurrentFlowCalculator, {"e_flow"}),
}


if __name__ == 'main':
    pass
