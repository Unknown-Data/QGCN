from measure_tests.run_specific_graph import load_graph, draw_graph
import networkx as nx

if __name__ == '__main__':
    path = 'undirected_test.pickle'
    # path = 'n_50_p_0.5_size_0'
    g = load_graph(path)
    G = nx.erdos_renyi_graph(10, 0.3, directed=True, seed=123453525)
    g = G.subgraph([0, 2, 3,6])
    draw_graph(g)
