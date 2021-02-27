import sys
import os

# Leave the path changes here!!!
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..','..','..'))

from src.accelerated_graph_features.test_python_converter import create_graph
from src.accelerated_graph_features.feature_wrappers import example_feature, clustering_coefficient, k_core, \
    node_page_rank, bfs_moments, motif, attraction_basin, flow

from graph_measures.loggers import PrintLogger
import numpy as np
from graph_measures.features_algorithms.vertices.motifs import MotifsNodeCalculator

import networkx as nx

from contextlib import contextmanager
from pprint import pprint


@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def pretify_motif_results(m_res, criteria=lambda x: x != 0):
    for node, node_list in enumerate(m_res[0]):
        print(node)
        node_string = ""
        for l in m_res:
            for k, v in enumerate(l[node]):
                if criteria(v):
                    node_string += '\t{}:{}'.format(k, int(v))
            node_string += '\n'
        print(node_string)


def test_specific():
    g = create_graph(3, GraphType=nx.DiGraph)
    #g = nx.gnp_random_graph(50, 0.5, directed=True, seed=123456)

    # print(attraction_basin(g))
    print(motif(g,level=3,gpu=True))


def test_features():
    g = create_graph(2, GraphType=nx.DiGraph)
    # print(attraction_basin(g))
    # print(flow(g))

    motif_level = 3
    # with silence_stdout():
    # example_feature(g)
    # clustering_coefficient(g)
    # k_core(g)
    # node_page_rank(g)
    # bfs_moments(g)
    # print(m_res)
    # gpu_motif = motif(g,level=motif_level,gpu=True)
    # print(len(gpu_motif),'X',len(gpu_motif[0]))
    # print(gpu_motif)
    # print(gpu_motif == m_res)
    # G = nx.random_regular_graph(20,100,seed=123456).to_directed()
    G = nx.erdos_renyi_graph(100, 0.5, directed=True, seed=None)
    # nx.write_gpickle(G,"test.pickle")
    # G = g
    # assert type(G) is nx.DiGraph
    logger = PrintLogger("My Logger")
    feature = MotifsNodeCalculator(G, level=motif_level, logger=logger)
    feature.build()
    mx = feature.to_matrix(mtype=np.matrix, should_zscore=False)
    # print(mx[0])
    m_res = motif(G, level=motif_level, gpu=False)
    # pretify_motif_results(m_res) 
    gpu_res = motif(G,level=motif_level,gpu=True)
  #  pretify_motif_results([ m_res,gpu_res])
    # pretify_motif_results([mx.tolist(),m_res,gpu_res])

    m_res_arr = np.asarray(m_res)
    gpu_res_arr = np.asarray(gpu_res)

    # print(mx.shape, m_res_arr.shape)

    #mx = gpu_res_arr
   # np_diff = mx - m_res_arr
    m_res_arr = gpu_res_arr
    np_diff = mx - m_res_arr
    print('diff avg:', np.average(np_diff[np.abs(np_diff) >= 1]))
    print('number of elements:', np.size(np_diff))
    print('number of different elements:', np.size(np_diff[np.abs(np_diff) >= 1]))
    print('Which is {}% of the elements'.format(100 * np.size(np_diff[np.abs(np_diff) >= 1]) / np.size(np_diff)))
    list_diff = []
    count = 0
    for l in np_diff:
        count += 1
        list_diff.append(l.tolist())
    #     print(list_diff[count-1])

    list_diff = [l[0] for l in list_diff]
 #   print(list_diff)
    #    print('count',count)
    pretify_motif_results([list_diff], lambda x: x != 0 and abs(x) > 1)

    avg = (mx + m_res_arr) / 2
    diff = (mx - m_res_arr)
    a = diff[avg != 0] / avg[avg != 0]
    a = a[np.abs(a) > 0.1]
    print(a)

    # print(len(gpu_res),'X',len(gpu_res[0]))
    # print(gpu_res[0])
    # print(gpu_res[1])


if __name__ == '__main__':
    test_features()
    #test_specific()
