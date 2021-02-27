import unittest
import numpy as np

import networkx as nx

from loggers import EmptyLogger
from measure_tests.test_graph import TestData, get_di_graph, get_graph

iterable_types = (list, tuple)
num_types = (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)


def compare_type(res1, res2):
    return (type(res1) == type(res2)) or \
           (isinstance(res1, iterable_types) and isinstance(res2, iterable_types)) or \
           (isinstance(res1, num_types) and isinstance(res2, num_types))


def are_results_equal(res1, res2, should_abs=False, ndigits=5):
    if res1 == res2:
        return True
    if not compare_type(res1, res2):
        return False
    if isinstance(res1, num_types):
        if should_abs:
            res1, res2 = abs(res1), abs(res2)
        if round(res1, ndigits) != round(res2, ndigits):
            return False
    elif isinstance(res1, dict):
        if res1.keys() != res2.keys():
            return False
        for key, val1 in res1.items():
            if not are_results_equal(val1, res2[key], should_abs=should_abs, ndigits=ndigits):
                return False
    elif isinstance(res1, iterable_types):
        if len(res1) != len(res2):
            return False
        for val1, val2 in zip(res1, res2):
            if not are_results_equal(val1, val2, should_abs=should_abs, ndigits=ndigits):
                return False
    else:
        assert False, "Unknown type"
    return True


def filter_gnx(gnx, is_max_connected=False):
    if not is_max_connected:
        return gnx
    if gnx.is_directed():
        subgraphs = nx.weakly_connected_component_subgraphs(gnx)
    else:
        subgraphs = nx.connected_component_subgraphs(gnx)
    return max(subgraphs, key=len)


class SpecificFeatureTest(unittest.TestCase):
    logger = EmptyLogger()

    @classmethod
    def setUpClass(cls):
        cls._test_data = TestData(logger=cls.logger)

    def _test_feature(self, feature_cls, is_directed, is_max_connected=True, manual=None, **cmp_features):
        gnx = get_di_graph() if is_directed else get_graph()
        gnx = filter_gnx(gnx, is_max_connected)
        feature = feature_cls(gnx, logger=self.logger)
        res = feature.build()
        # mx_res = feature.to_matrix()
        if manual is None:
            prev_res = self._test_data.load_feature(feature_cls, is_directed)
        else:
            prev_res = manual
        if prev_res is not None or feature.is_relevant():
            if not are_results_equal(res, prev_res, **cmp_features):
                are_results_equal(res, prev_res, **cmp_features)
            self.assertTrue(are_results_equal(res, prev_res, **cmp_features))


# def test_specific_feature(feature_cls):
#     self._test_feature(feature_cls, True)
#     self._test_feature(feature_cls, False)


if __name__ == '__main__':
    unittest.main()


# def _test_feature1(self, feature_cls, is_directed):
#     gnx = self._test_data.get_graph(is_directed=is_directed)
#     feature = feature_cls(gnx)
#     test_res = feature.build()
#     test_res = {item: [val] if not isinstance(val, list) else val for item, val in test_res.items()}
#
#     true_res = self._test_data.load_feature(feature_cls, is_directed=is_directed)
#     common = set(test_res).intersection(true_res)
#     for item in common:
#         for x, y in zip(test_res[item], true_res[item]):
#             self.assertAlmostEqual(x, y, 5)
#
#     test_diff = set(test_res).difference(true_res)
#     true_diff = set(true_res).difference(test_res)
