import os
import sys

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))


import networkx as nx
from graph_measures.loggers import PrintLogger
from pprint import pprint

from graph_measures.features_algorithms.vertices.motifs import MotifsNodeCalculator

logger = PrintLogger("Logger")


def create_graph(i=1, GraphType=nx.Graph):
    G = GraphType()
    if i == 1:
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)
        G.add_edge(2, 0)
        G.add_edge(3, 1)
        G.add_edge(3, 2)
    elif i == 2:
        G.add_edge(0, 1, weight=0.6)
        G.add_edge(0, 2, weight=0.2)
        G.add_edge(2, 3, weight=0.1)
        G.add_edge(2, 4, weight=0.7)
        G.add_edge(2, 5, weight=0.9)
        G.add_edge(0, 3, weight=0.3)

    elif i == 3:
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)
        G.add_edge(2, 0)
        G.add_edge(3, 1)
        G.add_edge(3, 2)
        G.add_edge(2, 4)
        G.add_edge(3, 4)
        G.add_edge(1, 5)
        G.add_edge(4, 5)
        G.add_edge(5, 6)
    return G


def compare_motifs(gnx, level=3):
    feature = MotifsNodeCalculator(gnx, level=level, logger=logger)
    feature.build()
    # pprint(feature._features)
    # mx = feature.to_matrix(should_zscore=False)
    # print(mx)


if __name__ == '__main__':
    gnx = create_graph(3,nx.Graph)
    compare_motifs(gnx, 3)
