from src.accelerated_graph_features.graph_converter import convert_graph_to_db_format
import networkx as nx


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


if __name__ == '__main__':
    indices, neighbors, w = convert_graph_to_db_format(create_graph(1, GraphType=nx.Graph), True, True)
    print(indices)
    print(neighbors)
    print(w)
