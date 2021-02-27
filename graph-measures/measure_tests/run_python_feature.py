import networkx as nx
from loggers import PrintLogger
import numpy as np
from features_algorithms.vertices.flow import FlowCalculator
from features_algorithms.vertices.motifs import MotifsNodeCalculator
import os
import matplotlib.pyplot as plt

logger = PrintLogger("MyLogger")


def build_graph():
    G = nx.DiGraph()
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


def main():
    g = build_graph()
    G = nx.erdos_renyi_graph(10, 0.3, directed=True, seed=123453525)
    g = G
    # pos = nx.spring_layout(g)
    # nx.draw(g,pos)
    # nx.draw_networkx_labels(g,pos)
    # plt.show()
    # feature = FlowCalculator(g)
    feature = MotifsNodeCalculator(g, level=4)
    feature.build()
    mx = feature.to_matrix(mtype=np.matrix, should_zscore=False)
    # print(mx)


if __name__ == '__main__':
    main()
