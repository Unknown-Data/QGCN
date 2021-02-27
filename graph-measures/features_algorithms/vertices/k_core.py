import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class KCoreCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        loopless_gnx = self._gnx.copy()
        loopless_gnx.remove_edges_from(nx.selfloop_edges(loopless_gnx))
        self._features = nx.core_number(loopless_gnx)


feature_entry = {
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
}

if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(KCoreCalculator, is_max_connected=True)
