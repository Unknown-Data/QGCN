import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

import networkx as nx
from loggers import PrintLogger
import numpy as np
from features_algorithms.vertices.motifs import MotifsNodeCalculator
import matplotlib.pyplot as plt

logger = PrintLogger("MyLogger")

PREFIX = 'specific_graphs'


def load_graph(path):
    g = nx.read_gpickle(open(os.path.join(PREFIX, path), 'rb'))
    center_node = 0
    # nodes = [center_node]
    # for i in range(3):
    #     addition = []
    #     for n in nodes:
    #         addition += list(g.neighbors(n))
    #     nodes += addition
    #
    # nodes = list(set(nodes))
    nodes = g.nodes

    # return nx.subgraph(g, nodes)
    return g


def draw_graph(gnx: nx.Graph):
    pos = nx.layout.spring_layout(gnx)
    nx.draw_networkx_nodes(gnx, pos)
    if gnx.is_directed():
        nx.draw_networkx_edges(gnx, pos, arrowstyle='->', arrowsize=30)
    else:
        nx.draw_networkx_edges(gnx, pos)

    nx.draw_networkx_labels(gnx, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show()


def main():
    path = 'undirected_test.pickle'
    # path = 'n_50_p_0.5_size_0'
    g = load_graph(path)
    # draw_graph(g)
    feature = MotifsNodeCalculator(g, level=4, logger=logger)
    feature.build()

    mx = feature.to_matrix(mtype=np.matrix, should_zscore=False)
    print(mx)
    pass


if __name__ == '__main__':
    main()
