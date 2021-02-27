from collections import Counter

import networkx as nx
import numpy as np


def original_bfs_moments(_gnx):
    _features = {}
    for node in _gnx:
        # calculate BFS distances
        distances = nx.single_source_shortest_path_length(_gnx, node)
        # distances.pop(node)
        # if not distances:
        #     self._features[node] = [0., 0.]
        #     continue
        node_dist = Counter(distances.values())
        dists, weights = zip(*node_dist.items())
        # This was in the previous version
        # instead of the above commented fix
        adjusted_dists = [x + 1 for x in dists]
        _features[node] = [float(np.average(weights, weights=adjusted_dists)), float(np.std(weights))]

    return _features


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(2, 0)
    G.add_edge(3, 1)
    G.add_edge(3, 2)
    print(original_bfs_moments(G))
