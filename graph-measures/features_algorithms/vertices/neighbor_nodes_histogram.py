import string
from functools import partial
from itertools import product as cartesian

import networkx as nx
import numpy as np

# Python 2, 3 compatible metaclass
# from future.utils import with_metaclass
# with_metaclass(SingletonName, object)
from scipy import sparse

from graph_measures.features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from graph_measures.loggers import PrintLogger

from datetime import datetime


class NthNeighborNodeHistogramCalculator(NodeFeatureCalculator):
    def __init__(self, neighbor_order, *args, **kwargs):
        super(NthNeighborNodeHistogramCalculator, self).__init__(*args, **kwargs)
        self.labels_to_consider = kwargs.get('labels_to_consider', self._gnx.graph["labels"])
        self._num_classes = len(self.labels_to_consider)
        self._neighbor_order = neighbor_order
        self._relation_types = ["".join(x) for x in cartesian(*(["io"] * self._neighbor_order))]
        self._print_name += "_%d" % (neighbor_order,)
        counter = {i: 0 for i in range(self._num_classes)}
        if self._gnx.is_directed():
            self._features = {node: {rtype: counter.copy() for rtype in self._relation_types} for node in self._gnx}
        else:
            self._features = {node: counter.copy() for node in self._gnx}

    def is_relevant(self):
        is_empty = True if self._neighbor_order == 0 else False
        return not is_empty
        # # undirected is not supported yet
        # return self._gnx.is_directed() and not is_empty

    def _get_node_neighbors_with_types(self, node):
        if self._gnx.is_directed():
            for in_edge in self._gnx.in_edges(node):
                yield ("i", in_edge[0])

            for out_edge in self._gnx.out_edges(node):
                yield ("o", out_edge[1])
        else:
            for edge in self._gnx.edges(node):
                yield (edge[1])

    def _iter_nodes_of_order(self, node, order: int):
        if self._gnx.is_directed():
            if 0 >= order:
                yield [], node
                return
            for r_type, neighbor in self._get_node_neighbors_with_types(node):
                for r_type2, neighbor2 in self._iter_nodes_of_order(neighbor, order - 1):
                    yield ([r_type] + r_type2, neighbor2)
        else:
            if 0 >= order:
                yield node
                return
            for neighbor in self._get_node_neighbors_with_types(node):
                for neighbor2 in self._iter_nodes_of_order(neighbor, order - 1):
                    yield (neighbor2)

    def _calculate(self, include: set):

        # Translating each label to a relevant index to save memory
        labels_map = {label: idx for idx, label in enumerate(self.labels_to_consider)}

        if self._gnx.is_directed():
            # self._calculate_directed2(include, labels_map)
            self._calculate_directed2(include, labels_map)
        else:
            # self._calculate_undirected(include, labels_map)
            self._calculate_undirected2(include, labels_map)

    def _calculate_directed(self, include: set, labels_map):
        for node in self._gnx:
            history = {rtype: set() for rtype in self._relation_types}
            for r_type, neighbor in self._iter_nodes_of_order(node, self._neighbor_order):
                full_type = "".join(r_type)
                if node == neighbor or neighbor not in include or neighbor in history[full_type]:
                    continue
                history[full_type].add(neighbor)

                # in case the label is already the index of the label in the labels_map
                neighbor_color = self._gnx.node[neighbor]["label"]
                if neighbor_color in labels_map:
                    neighbor_color = labels_map[neighbor_color]
                self._features[node][full_type][neighbor_color] += 1

    def _calculate_directed2(self, include: set, labels_map):
        for node in include:
            # in case the label is already the index of the label in the labels_map
            color = self._gnx.node[node]["label"]
            if color in labels_map:
                color = labels_map[color]
            else:
                continue
            history = {rtype: set() for rtype in self._relation_types}
            for r_type, neighbor in self._iter_nodes_of_order(node, self._neighbor_order):
                full_type = "".join(r_type)
                neighbor_color = self._gnx.node[node]["label"]
                if node == neighbor or neighbor in history[full_type] or neighbor_color not in labels_map:
                    continue
                history[full_type].add(neighbor)

                self._features[neighbor][full_type][color] += 1

    def _calculate_undirected(self, include: set, labels_map):
        for node in self._gnx:
            history = set()
            for neighbor in self._iter_nodes_of_order(node, self._neighbor_order):
                if node == neighbor or neighbor not in include or neighbor in history:
                    continue
                history.add(neighbor)

                # in case the label is already the index of the label in the labels_map
                neighbor_color = self._gnx.node[neighbor]["label"]
                if neighbor_color in labels_map:
                    neighbor_color = labels_map[neighbor_color]
                self._features[node][neighbor_color] += 1

    def _calculate_undirected2(self, include: set, labels_map):
        for node in include:
            # in case the label is already the index of the label in the labels_map
            color = self._gnx.node[node]["label"]
            if color in labels_map:
                color = labels_map[color]
            history = set()
            for neighbor in self._iter_nodes_of_order(node, self._neighbor_order):
                if node == neighbor or neighbor in history:
                    continue
                history.add(neighbor)

                self._features[neighbor][color] += 1

    def _get_feature(self, element):
        cur_feature = self._features[element]
        if not self._gnx.is_directed():
            return np.array([cur_feature[x] for x in range(self._num_classes)])

        return np.array([[cur_feature[r_type][x] for x in range(self._num_classes)]
                         for r_type in self._relation_types]).flatten()

    # def _to_ndarray(self):
    #     mx = np.matrix([self._get_feature(node) for node in self._nodes()])
    #     return mx.astype(np.float32)


def nth_neighbor_calculator(order, **kwargs):
    # Kwargs are used to pass the labels to consider argument
    return partial(NthNeighborNodeHistogramCalculator, order, **kwargs)


feature_entry = {
    "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}


def build_sample_graph(edges, colors, color_list):
    dg = nx.DiGraph(labels=color_list)  # list(set(colors.values())))
    dg.add_edges_from(edges)
    for n in dg:
        dg.node[n]['label'] = colors[n]
    return dg


def test_neighbor_histogram():
    all_colors = ['red', 'blue', 'green', 'yellow']
    colors = {name: i for i, name in enumerate(all_colors)}
    node_colors = {x: all_colors[i % len(all_colors)] for i, x in enumerate(string.ascii_letters)}
    gnx = build_sample_graph([('a', 'b'), ('b', 'c'), ('a', 'd'), ('c', 'd'), ('e', 'f'), ('f', 'a')], node_colors,
                             all_colors)
    logger = PrintLogger()
    calc = NthNeighborNodeHistogramCalculator(2, gnx, logger=logger, labels_to_consider=all_colors[:3])
    calc._calculate(include=set(gnx.nodes))
    # n = calc._to_ndarray()
    # (self, gnx, name, abbreviations, logger=None):
    # m = calculate_second_neighbor_vector(gnx, colors)
    print('bla')


if __name__ == "__main__":
    test_neighbor_histogram()
