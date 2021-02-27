from functools import partial
from itertools import product as cartesian

import networkx as nx
import numpy as np

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from loggers import PrintLogger


# Python 2, 3 compatible metaclass
# from future.utils import with_metaclass
# with_metaclass(SingletonName, object)


class NthNeighborNodeEdgeHistogramCalculator(NodeFeatureCalculator):
    def __init__(self, neighbor_order, *args, **kwargs):
        super(NthNeighborNodeEdgeHistogramCalculator, self).__init__(*args, **kwargs)
        # self._num_classes = len(self._gnx.graph["node_labels"])
        self._neighbor_order = neighbor_order
        self._relation_types = ["".join(x) for x in cartesian(*(["io"] * self._neighbor_order))]
        self._print_name += "_%d" % (neighbor_order,)
        counter = {i: 0 for i in self._gnx.graph["edge_labels"]}
        self._features = {node: {rtype: counter.copy() for rtype in self._relation_types} for node in self._gnx}
        # self._counter_tuple = namedtuple("EdgeCounter", self._relation_types)

    def is_relevant(self):
        return self._gnx.is_directed()

    def _get_node_neighbors_with_types(self, node):
        if self._gnx.is_directed():
            for edge in self._gnx.in_edges(node, data=True):
                yield (edge, edge[0], "i")  # , {"type": "i", "data": data})

        # By default, edges are out_edges for directed graphs
        for edge in self._gnx.edges(node, data=True):
            yield (edge, edge[1], "o")  # n2, {"type": "o", "data": data})

    def _iter_edges_of_order(self, node, order: int):
        for edge, neighbor, r_type1 in self._get_node_neighbors_with_types(node):
            if order <= 1:
                yield (edge, neighbor, [r_type1])
                continue

            for edge2, neighbor2, r_type2 in self._iter_edges_of_order(neighbor, order - 1):
                yield (edge2, neighbor2, [r_type1] + r_type2)

    def _calculate(self, include: set):
        # Translating each label to a relevant index to save memory
        # labels_map = {label: idx for idx, label in enumerate(self._gnx.graph["node_labels"])}

        for node in self._gnx:
            history = {rtype: set() for rtype in self._relation_types}
            for edge, neighbor, r_type in self._iter_edges_of_order(node, self._neighbor_order):
                full_type = "".join(r_type)  # map(itemgetter("type"), meta))
                if node == neighbor or neighbor in history[full_type]:  # neighbor not in include or
                    continue
                history[full_type].add(edge[:2])

                # in case the label is already the index of the label in the labels_map
                # neighbor_color = self._gnx.node[neighbor]["label"]
                # if neighbor_color in labels_map:
                #     neighbor_color = labels_map[neighbor_color]
                neighbor_color = edge[2]["label"]
                self._features[node][full_type][neighbor_color] += 1

    def _get_feature(self, element):
        return np.array([[self._features[element][r_type][x] for x in self._gnx.graph["edge_labels"]]
                         for r_type in self._relation_types]).flatten()

    # def _to_ndarray(self):
    #     mx = np.matrix([self._get_feature(node) for node in self._nodes()])
    #     return mx.astype(np.float32)


def nth_neighbor_calculator(order):
    return partial(NthNeighborNodeEdgeHistogramCalculator, order)


feature_entry = {
    "first_node_edge_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnneh"}),
    "second_node_edge_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snneh"}),
}


def sample_graph():
    g = nx.DiGraph(edge_labels=[1, 2])
    g.add_edges_from([
        (1, 2, {"label": 1}),
        (2, 6, {"label": 2}),
        (4, 1, {"label": 1}),
        (4, 5, {"label": 2}),
        (5, 2, {"label": 1}),
        (5, 3, {"label": 2}),
        (6, 4, {"label": 1}),
    ])
    return g


def test_neighbor_histogram():
    gnx = sample_graph()
    logger = PrintLogger()
    calc = NthNeighborNodeEdgeHistogramCalculator(2, gnx, logger=logger)
    calc.build()
    n = calc.to_matrix()
    # (self, gnx, name, abbreviations, logger=None):
    # m = calculate_second_neighbor_vector(gnx, colors)
    print('bla')


if __name__ == "__main__":
    test_neighbor_histogram()
