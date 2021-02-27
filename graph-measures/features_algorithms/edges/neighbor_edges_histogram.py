from itertools import product as cartesian

import networkx as nx
import numpy as np

from features_infra.feature_calculators import EdgeFeatureCalculator
from loggers import PrintLogger


# Python 2, 3 compatible metaclass
# from future.utils import with_metaclass
# with_metaclass(SingletonName, object)


class NeighborEdgeHistogramCalculator(EdgeFeatureCalculator):
    def __init__(self, *args, **kwargs):
        super(NeighborEdgeHistogramCalculator, self).__init__(*args, **kwargs)
        self._num_classes = len(self._gnx.graph["edge_labels"])
        self._relation_types = ["".join(x) for x in cartesian("io", "io")]
        counter = {i: 0 for i in range(self._num_classes)}
        self._features = {edge: {rtype: counter.copy() for rtype in self._relation_types} for edge in self._gnx.edges()}

    def is_relevant(self):
        return self._gnx.is_directed()

    def _neighbor_edges(self, node):
        for in_edge in self._gnx.in_edges(node):
            yield ("i", in_edge)

        for out_edge in self._gnx.out_edges(node):
            yield ("o", out_edge)

    def _iter_neighbor_edges(self, edge):
        for etype, e in self._neighbor_edges(edge[0]):
            yield (["o", etype], e)

        for etype, e in self._neighbor_edges(edge[1]):
            yield (["i", etype], e)

    def _calculate(self, include: set):
        # Translating each label to a relevant index to save memory
        labels_map = {label: idx for idx, label in enumerate(self._gnx.graph["edge_labels"])}

        for i, edge in enumerate(self._gnx.edges()):
            history = {rtype: set() for rtype in self._relation_types}
            for r_type, neighbor in self._iter_neighbor_edges(edge):
                full_type = "".join(r_type)
                if edge == neighbor or neighbor in history[full_type]:  # neighbor not in include or
                    continue
                history[full_type].add(neighbor)

                # in case the label is already the index of the label in the labels_map
                neighbor_color = self._gnx.edges[neighbor]["label"]
                if neighbor_color in labels_map:
                    neighbor_color = labels_map[neighbor_color]
                self._features[edge][full_type][neighbor_color] += 1

            if 0 == i % 1000:
                self._logger.debug("Processed %d edges", i)
            if i == 8000:
                self._logger.debug("Stopping for processing")
                break

    def _get_feature(self, element):
        return np.array([[self._features[element][r_type][x] for x in range(self._num_classes)]
                         for r_type in self._relation_types]).flatten()

    # def _to_ndarray(self):
    #     mx = np.matrix([self._get_feature(node) for node in self._nodes()])
    #     return mx.astype(np.float32)


def build_sample_graph():
    dg = nx.DiGraph(edge_labels=[-1, 0, 1])
    dg.add_edges_from([
        (1, 2, {"label": -1}),
        (1, 3, {"label": -1}),
        (2, 3, {"label": 1}),
        (4, 3, {"label": 0}),
    ])
    return dg


# def nth_neighbor_calculator(order):
#     return partial(NthNeighborHistogramCalculator, order)


def test_neighbor_histogram():
    import pickle
    logger = PrintLogger()
    gnx = pickle.load(open("/home/orion/git/data/epinions/gnx.pkl", "rb"))
    logger.info("Graph loaded")
    calc = NeighborEdgeHistogramCalculator(gnx, logger=logger)
    calc.build()
    print("Bla")


def test_neighbor_histogram_old():
    gnx = build_sample_graph()
    logger = PrintLogger()
    calc = NeighborEdgeHistogramCalculator(gnx, logger=logger)
    calc.build()
    print('bla')


if __name__ == "__main__":
    test_neighbor_histogram()
