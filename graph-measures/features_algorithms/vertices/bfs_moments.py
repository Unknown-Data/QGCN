from collections import Counter

import networkx as nx
import numpy as np

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class BfsMomentsCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def weighted_avg_and_std(self, values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        return average, np.sqrt(variance)

    def _calculate(self, include: set):
        for node in self._gnx:
            # calculate BFS distances
            distances = nx.single_source_shortest_path_length(self._gnx, node)
            # distances.pop(node)
            # if not distances:
            #     self._features[node] = [0., 0.]
            #     continue
            node_dist = Counter(distances.values())
            dists, weights = zip(*node_dist.items())
            # This was in the previous version
            # instead of the above commented fix
            adjusted_dists = np.asarray([x + 1 for x in dists])
            weights = np.asarray(weights)
            self._features[node] = [self.weighted_avg_and_std(adjusted_dists, weights)]

    def _get_feature(self,element):
        return list(self._features[element])[0]

feature_entry = {
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
}

if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature

    test_specific_feature(BfsMomentsCalculator, is_max_connected=True)
