import networkx as nx

from features_infra.feature_calculators import EdgeFeatureCalculator, FeatureMeta


class EdgeBetweennessCalculator(EdgeFeatureCalculator):
    def _calculate(self, include: set):
        self._features = nx.edge_betweenness_centrality(self._gnx)

    def is_relevant(self):
        return True


feature_entry = {
    "edge_betweenness": FeatureMeta(EdgeBetweennessCalculator, {"e_bet"}),
}

if __name__ == 'main':
    pass
