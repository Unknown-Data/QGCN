import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths import weighted

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class FlowCalculator(NodeFeatureCalculator):
    """See Y. Rozen & Y. Louzoun article <add-link>"""

    def __init__(self, *args, threshold=0, **kwargs):
        super(FlowCalculator, self).__init__(*args, **kwargs)
        self._threshold = threshold

    def is_relevant(self):
        return self._gnx.is_directed()

    def _calculate(self, threshold):
        num_nodes = len(self._gnx)
        directed_dists = dict(weighted.all_pairs_dijkstra_path_length(self._gnx, num_nodes, weight='weight'))
        undirected_dists = dict(
            weighted.all_pairs_dijkstra_path_length(self._gnx.to_undirected(), num_nodes, weight='weight'))

        # calculate the number of nodes reachable to/ from node 'n'
        b_u = {node: len(set(nx.ancestors(self._gnx, node)).union(nx.descendants(self._gnx, node)))
               for node in self._gnx}
        max_b_u = float(max(b_u.values()))

        for node in self._gnx:
            # the delta determines whether this node is to be considered
            if (b_u[node] / max_b_u) <= self._threshold:
                self._features[node] = 0
                continue

            udists = undirected_dists[node]
            dists = directed_dists[node]

            # getting coordinated values from two dictionaries with the same keys
            # saving the data as np.array type
            num, denom = map(np.array, zip(*((udists[n], dists[n]) for n in dists)))

            num = num[denom != 0]
            denom = denom[denom != 0]

            self._features[node] = np.sum(num / denom) / float(b_u[node])


feature_entry = {
    "flow": FeatureMeta(FlowCalculator, {}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(FlowCalculator, is_max_connected=True)
