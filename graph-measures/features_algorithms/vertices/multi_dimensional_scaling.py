import networkx as nx
import numpy as np
from sklearn.manifold import Isomap

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


MAX_DEGREE = 5
COMPONENT_SIZE = 20


class MultiDimensionalScalingCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        self._features = {}
        for graph in nx.connected_component_subgraphs(self._gnx.to_undirected()):
            nodes_order = sorted(graph)
            dissimilarities = self._dissimilarity(graph, nodes_order)
            min_degrees = min(graph.degree(), key=lambda x: x[1])[1]  # [deg for node, deg in graph.degree()])
            isomap_mx = Isomap(n_neighbors=min(min_degrees, MAX_DEGREE),
                               n_components=COMPONENT_SIZE).fit_transform(dissimilarities)
            self._features.update(zip(nodes_order, isomap_mx))

    @staticmethod
    def _dissimilarity(graph, nodes_order):
        m = nx.floyd_warshall_numpy(graph, nodelist=nodes_order)
        return np.asarray(m)


feature_entry = {
    "multi_dimensional_scaling": FeatureMeta(MultiDimensionalScalingCalculator, {"mds"}),
}


def test_feature():
    from loggers import PrintLogger
    from measure_tests.test_graph import get_graph
    gnx = get_graph()
    feat = MultiDimensionalScalingCalculator(gnx, logger=PrintLogger("Keren's Logger"))
    res = feat.build()
    print(res)


if __name__ == "__main__":
    # from measure_tests.specific_feature_test import test_specific_feature
    # test_specific_feature(MultiDimensionalScaling, is_max_connected=True)
    test_feature()
