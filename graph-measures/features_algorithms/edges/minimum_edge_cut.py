import networkx as nx

from features_infra.feature_calculators import EdgeFeatureCalculator, FeatureMeta


class MinimumEdgeCutCalculator(EdgeFeatureCalculator):

    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        self._features = nx.minimum_edge_cut(self._gnx)


feature_entry = {
    "minimum_edge_cut": FeatureMeta(MinimumEdgeCutCalculator, {"min_cut"}),
}

if __name__ == 'main':
    pass
