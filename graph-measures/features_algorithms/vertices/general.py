from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class GeneralCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        if self._gnx.is_directed():
            self._features = {node: (in_deg, out_deg) for
                              (node, out_deg), (_, in_deg) in zip(self._gnx.out_degree(), self._gnx.in_degree())}
        else:
            self._features = {node: deg for node, deg in self._gnx.degree()}


feature_entry = {
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(GeneralCalculator, is_max_connected=True)
