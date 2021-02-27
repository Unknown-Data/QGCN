import numpy as np

from features_algorithms.vertices.general import GeneralCalculator
from features_infra.feature_calculators import EdgeFeatureCalculator, FeatureMeta


class EdgeDegreeBasedCalculator(EdgeFeatureCalculator):
    def _calculate(self, include: set):
        self._general_c = GeneralCalculator(self._gnx, self._logger)
        self._general_c.build()
        if self._gnx.is_directed():
            self._edge_based_degree_directed()
        else:
            self._edge_based_degree_undirected()

    def is_relevant(self):
        return True

    def _edge_based_degree_directed(self):
        for edge in self._gnx.edges():
            e1_feature = np.array(self._general_c.feature(edge[0]))
            e2_feature = np.array(self._general_c.feature(edge[1]))

            self._features[edge] = np.concatenate([
                (e1_feature - e2_feature).astype(np.float32),  # sub out-in
                np.mean([e1_feature, e2_feature], axis=1).astype(np.float32),  # mean out-in
            ])

    def _edge_based_degree_undirected(self):
        for edge in self._gnx.edges():
            e1_feature = self._general_c.feature(edge[0])
            e2_feature = self._general_c.feature(edge[1])

            self._features[edge] = [
                float(e1_feature[0]) - e2_feature[0],  # sub
                (float(e1_feature[0]) + e2_feature[0]) / 2  # mean
            ]


feature_entry = {
    "Edge_degree_based_calculator": FeatureMeta(EdgeDegreeBasedCalculator, {"e_degree"}),
}

if __name__ == 'main':
    pass
