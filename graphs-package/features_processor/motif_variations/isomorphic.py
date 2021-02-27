import json
import pickle
from itertools import combinations, permutations

import networkx as nx


class IsomorphismGenerator:
    def __init__(self, group_size, is_directed):
        self._group_size = group_size
        self._is_directed = is_directed
        graphs = self._generate_all_graphs()
        self._isomorphisms = self._group_to_isomorphisms(graphs)
        self._remove_irrelevant()
        self._reorganize()

    # Generate all possible graphs of size 'group_size'
    def _generate_all_graphs(self):
        # handshake lemma
        num_edges = int(self._group_size * (self._group_size - 1) / 2.)
        num_bits = num_edges * 2 if self._is_directed else num_edges
        edge_iter = permutations if self._is_directed else combinations
        graph_type = nx.DiGraph if self._is_directed else nx.Graph

        graphs = {}
        for num in range(2 ** num_bits):
            g = graph_type()
            g.add_nodes_from(range(self._group_size))
            g.add_edges_from((x, y) for i, (x, y) in enumerate(edge_iter(range(self._group_size), 2)) if (2 ** i) & num)
            graphs[num] = g
        return graphs

    @staticmethod
    def _group_to_isomorphisms(graphs):
        isomorphisms = {}
        i = 0
        keys = sorted(list(graphs.keys()))
        while keys:
            g1 = graphs[keys[0]]
            isomorphisms[i] = {num: graphs[num] for num in keys if nx.is_isomorphic(g1, graphs[num])}
            keys = [x for x in keys if x not in isomorphisms[i]]
            i += 1
        return isomorphisms

    def _remove_irrelevant(self):
        isomorphisms = self._isomorphisms
        # Remove disconnected graphs
        irrelevant = [n for n, gs in isomorphisms.items() if not nx.is_connected(list(gs.values())[0].to_undirected())]
        if irrelevant:
            isomorphisms[None] = {}
        for n in irrelevant:
            isomorphisms[None].update(isomorphisms.pop(n))

    def _reorganize(self):
        keys = [x for x in self._isomorphisms if x is not None]
        irrelevant = self._isomorphisms.get(None)
        self._isomorphisms = {i: self._isomorphisms[key] for i, key in enumerate(keys)}
        if irrelevant:
            self._isomorphisms[None] = irrelevant

    def num_2_motif(self):
        return {num: motif_num for motif_num, group in self._isomorphisms.items() for num in group}


def main(level, is_directed):
    fname = "%d_%sdirected" % (level, "" if is_directed else "un")
    print("Calculating ", fname)
    gs = IsomorphismGenerator(level, is_directed)
    # Json dump integers to strings (JavaScript compatibility), other option - to dump
    json.dump(list(gs.num_2_motif().items()), open(fname + ".json", "w"))
    pickle.dump(gs.num_2_motif(), open(fname + ".pkl", "wb"))
    print("Finished calculating ", fname)
    # for y in gs.values():
    #     print(list(map(lambda i: len(i.edges()), y.values())))


if __name__ == "__main__":
    main(3, False)
    main(3, True)
    main(4, False)
    main(4, True)
    # main(5, False)
    # main(5, True)
    print("Bla")
