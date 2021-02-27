from collections import Counter

import community

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class LouvainCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        # relevant only for undirected graphs
        return not self._gnx.is_directed()

    def _calculate(self, include: set):
        partition = community.best_partition(self._gnx)
        com_size_dict = Counter(partition.values())
        self._features = {node: com_size_dict[partition[node]] for node in self._gnx}


feature_entry = {
    "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
}

if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(LouvainCalculator, is_max_connected=True)
