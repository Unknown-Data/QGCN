from collections import Counter

import networkx as nx
from networkx.algorithms.shortest_paths import weighted

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class AttractorBasinCalculator(NodeFeatureCalculator):
    def __init__(self, *args, alpha=2, **kwargs):
        super(AttractorBasinCalculator, self).__init__(*args, **kwargs)
        self._alpha = alpha
        self._default_val = float('nan')

    def is_relevant(self):
        return self._gnx.is_directed()

    def _initialize_attraction_basin_dist(self):
        ab_in_dist = {}
        ab_out_dist = {}

        # for each node we are calculating the the out and in distances for the other nodes in the graph
        dists = dict(weighted.all_pairs_dijkstra_path_length(self._gnx, len(self._gnx), weight='weight'))
        for node in self._gnx:
            if node not in dists:
                continue

            node_dists = dists[node]
            ab_out_dist[node] = Counter([node_dists[d] for d in nx.descendants(self._gnx, node)])
            ab_in_dist[node] = Counter([dists[d][node] for d in nx.ancestors(self._gnx, node)])

        return ab_out_dist, ab_in_dist

    def _calculate(self, incluse: set):
        ab_out_dist, ab_in_dist = self._initialize_attraction_basin_dist()
        avg_out = self._calculate_average_per_dist(len(self._gnx), ab_out_dist)
        avg_in = self._calculate_average_per_dist(len(self._gnx), ab_in_dist)

        # running on all the nodes and calculate the value of 'attraction_basin'
        for node in self._gnx:
            out_dist = ab_out_dist.get(node, {})
            in_dist = ab_in_dist.get(node, {})

            self._features[node] = self._default_val
            denominator = sum((dist / avg_out[m]) * (self._alpha ** (-m)) for m, dist in out_dist.items())
            if 0 != denominator:
                numerator = sum((dist / avg_in[m]) * (self._alpha ** (-m)) for m, dist in in_dist.items())
                self._features[node] = numerator / denominator

    @staticmethod
    def _calculate_average_per_dist(num_nodes, count_dist):
        # rearrange the details in "count_dist" to be with unique distance in the array "all_dist_count"
        all_dist_count = {}
        for counter in count_dist.values():
            for dist, occurrences in counter.items():
                all_dist_count[dist] = all_dist_count.get(dist, 0) + occurrences

        # calculating for each distance the average
        return {dist: float(count) / num_nodes for dist, count in all_dist_count.items()}


feature_entry = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(AttractorBasinCalculator, is_max_connected=True)
