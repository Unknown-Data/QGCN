import os
import pickle
from functools import partial
from itertools import permutations, combinations

import networkx as nx
import numpy as np
from bitstring import BitArray
from collections import Counter

try:
    from graph_measures.features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
except ModuleNotFoundError as e:
    from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta

CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(os.path.dirname(CUR_PATH))
VERBOSE = False

DEBUG =False 
SAVE_COUNTED_MOTIFS = False

interesting_groups = [
    sorted([0, 1, 8, 27])

]


class MotifsNodeCalculator(NodeFeatureCalculator):
    def __init__(self, *args, level=3, **kwargs):
        super(MotifsNodeCalculator, self).__init__(*args, **kwargs)
        assert level in [3, 4], "Unsupported motif level %d" % (level,)
        self._level = level
        self._node_variations = {}
        self._all_motifs = None
        self._print_name += "_%d" % (self._level,)
        self._gnx = self._gnx.copy()
        self._load_variations()

        self._counted_motifs = set()  # Only used if SAVE_COUNTED_MOTIFS is set
        self._double_counter = Counter()

    def is_relevant(self):
        return True

    @classmethod
    def print_name(cls, level=None):
        print_name = super(MotifsNodeCalculator, cls).print_name()
        if level is None:
            return print_name
        return "%s_%d" % (print_name, level)
        # name = super(MotifsNodeCalculator, cls).print_name()
        # name.split("_")[0]

    def _load_variations_file(self):
        fname = "%d_%sdirected.pkl" % (self._level, "" if self._gnx.is_directed() else "un")
        fpath = os.path.join(BASE_PATH, "motif_variations", fname)
        return pickle.load(open(fpath, "rb"))

    def _load_variations(self):
        self._node_variations = self._load_variations_file()
        self._all_motifs = set(self._node_variations.values())

    # here we pass on the edges of the sub-graph containing only the bunch nodes
    # and calculate the expected index of each edge (with respect to whether the graph is directed on not)
    # the formulas were calculated by common reason
    # combinations index: sum_0_to_n1-1((n - i) - 1) + n2 - n1 - 1
    # permutations index: each set has (n - 1) items, so determining the set is by n1, and inside the set by n2
    def _get_group_number_opt1(self, nbunch):
        subgnx = self._gnx.subgraph(nbunch)
        nodes = {node: i for i, node in enumerate(subgnx)}
        n = len(nodes)

        if subgnx.is_directed():
            def edge_index(n1, n2):
                return n1 * (n - 1) + n2 - (1 * (n2 > n1))
        else:
            def edge_index(n1, n2):
                n1, n2 = min(n1, n2), max(n1, n2)
                return (n1 / 2) * (2 * n - 3 - n1) + n2 - 1

        return sum(2 ** edge_index(nodes[edge[0]], nodes[edge[1]]) for edge in subgnx.edges())

    # passing on all:
    #  * undirected graph: combinations [(n*(n-1)/2) combs - handshake lemma]
    #  * directed graph: permutations [(n*(n-1) perms - handshake lemma with respect to order]
    # checking whether the edge exist in the graph - and construct a bitmask of the existing edges
    def _get_group_number(self, nbunch):
        func = permutations if self._gnx.is_directed() else combinations
        if DEBUG:
            pass
        return BitArray(self._gnx.has_edge(n1, n2) for n1, n2 in func(nbunch, 2)).uint

    # def _get_motif_sub_tree(self, root, length):

    # implementing the "Kavosh" algorithm for subgroups of length 3
    def _get_motif3_sub_tree(self, root):
        visited_vertices = {root: 0}
        visited_index = 1

        # variation == (1, 1)
        first_neighbors = set(nx.all_neighbors(self._gnx, root))
        # neighbors, visited_neighbors = tee(first_neighbors)
        for n1 in first_neighbors:
            visited_vertices[n1] = visited_index
            visited_index += 1

        for n1 in first_neighbors:
            last_neighbors = set(nx.all_neighbors(self._gnx, n1))
            for n2 in last_neighbors:
                if n2 in visited_vertices:
                    if visited_vertices[n1] < visited_vertices[n2]:
                        yield [root, n1, n2]
                else:
                    visited_vertices[n2] = visited_index
                    visited_index += 1
                    yield [root, n1, n2]
        # variation == (2, 0)
        for n1, n2 in combinations(first_neighbors, 2):
            if (visited_vertices[n1] < visited_vertices[n2]) and \
                    not (self._gnx.has_edge(n1, n2) or self._gnx.has_edge(n2, n1)):
                yield [root, n1, n2]

    # implementing the "Kavosh" algorithm for subgroups of length 4
    def _get_motif4_sub_tree(self, root):
        visited_vertices = {root: 0}
        # visited_index = 1

        # variation == (1, 1, 1)
        neighbors_first_deg = set(nx.all_neighbors(self._gnx, root))
        # neighbors_first_deg, visited_neighbors, len_a = tee(neighbors_first_deg, 3)
        neighbors_first_deg = visited_neighbors = list(neighbors_first_deg)

        for n1 in visited_neighbors:
            visited_vertices[n1] = 1
        for n1, n2, n3 in combinations(neighbors_first_deg, 3):
            group = [root, n1, n2, n3]
            if DEBUG:

                if sorted(group) in interesting_groups:
                    print('An interesting group:', group)

            yield group

        for n1 in neighbors_first_deg:
            if DEBUG:
                pass

            neighbors_sec_deg = set(nx.all_neighbors(self._gnx, n1))
            # neighbors_sec_deg, visited_neighbors, len_b = tee(neighbors_sec_deg, 3)
            neighbors_sec_deg = visited_neighbors = list(neighbors_sec_deg)
            for n in visited_neighbors:
                if n not in visited_vertices:
                    if DEBUG:
                        if n is 1:
                            hi = 0.5
                    visited_vertices[n] = 2
            for n2 in neighbors_sec_deg:
                for n11 in neighbors_first_deg:
                    if visited_vertices[n2] == 2 and n1 != n11:
                        edge_exists = (self._gnx.has_edge(n2, n11) or self._gnx.has_edge(n11, n2))
                        if (not edge_exists) or (edge_exists and n1 < n11):

                            group = [root, n1, n11, n2]
                            if DEBUG:
                                if sorted(group) in interesting_groups:
                                    print('An interesting group:', group)

                            yield group
            # for n1 in neighbors_first_deg:
            #     if DEBUG:
            #         if root is 41:
            #             print('n1', n1)
            #     neighbors_sec_deg = set(nx.all_neighbors(self._gnx, n1))
            #     # neighbors_sec_deg, visited_neighbors, len_b = tee(neighbors_sec_deg, 3)
            #     neighbors_sec_deg = visited_neighbors = list(neighbors_sec_deg)

            for comb in combinations(neighbors_sec_deg, 2):
                if DEBUG:
                    if root is 41:
                        hi = 1
                if 2 == visited_vertices[comb[0]] and visited_vertices[comb[1]] == 2:
                    group = [root, n1, comb[0], comb[1]]
                    if DEBUG:
                        if root is 41:
                            print('A 41 group:', group)
                        if sorted(group) in interesting_groups:
                            print('An interesting group:', group)

                    yield group

        for n1 in neighbors_first_deg:
            if DEBUG:
                pass

            neighbors_sec_deg = set(nx.all_neighbors(self._gnx, n1))
            # neighbors_sec_deg, visited_neighbors, len_b = tee(neighbors_sec_deg, 3)
            neighbors_sec_deg = visited_neighbors = list(neighbors_sec_deg)
            for n2 in neighbors_sec_deg:
                if visited_vertices[n2] == 1:
                    continue

                for n3 in set(nx.all_neighbors(self._gnx, n2)):
                    if DEBUG:
                        if root is 0 and n1 is 27 and n2 is 8 and n3 is 1:
                            hi = 1.5

                    if n3 not in visited_vertices:
                        if DEBUG:
                            pass

                        visited_vertices[n3] = 3
                        if visited_vertices[n2] == 2:
                            group = [root, n1, n2, n3]
                            if DEBUG:
                                if sorted(group) in interesting_groups:
                                    print('An interesting group:', group)

                            yield group
                    else:

                        if visited_vertices[n3] == 1:
                            continue

                        if visited_vertices[n3] == 2 and not (self._gnx.has_edge(n1, n3) or self._gnx.has_edge(n3, n1)):
                            group = [root, n1, n2, n3]
                            if DEBUG:
                                if sorted(group) in interesting_groups:
                                    print('An interesting group:', group)

                            yield group

                        elif visited_vertices[n3] == 3 and visited_vertices[n2] == 2:

                            group = [root, n1, n2, n3]
                            if DEBUG:
                                if sorted(group) in interesting_groups:
                                    print('An interesting group:', group)

                            yield group

    def _order_by_degree(self, gnx=None):
        if gnx is None:
            gnx = self._gnx
        return sorted(gnx, key=lambda n: len(list(nx.all_neighbors(gnx, n))), reverse=True)

    def _calculate_motif(self):
        # consider first calculating the nth neighborhood of a node
        # and then iterate only over the corresponding graph
        motif_func = self._get_motif3_sub_tree if self._level == 3 else self._get_motif4_sub_tree
        sorted_nodes = self._order_by_degree()
        for node in sorted_nodes:
            for group in motif_func(node):
                group_num = self._get_group_number(group)
                motif_num = self._node_variations[group_num]
                yield group, group_num, motif_num
            if VERBOSE:
                self._logger.debug("Finished node: %s" % node)
            self._gnx.remove_node(node)

    def _update_nodes_group(self, group, motif_num):
       for node in group:
            self._features[node][motif_num] += 1

    def _calculate(self, include=None):
        m_gnx = self._gnx.copy()
        motif_counter = {motif_number: 0 for motif_number in self._all_motifs}
        self._features = {node: motif_counter.copy() for node in self._gnx}
        for i, (group, group_num, motif_num) in enumerate(self._calculate_motif()):
            if DEBUG:

                if 21 in group and motif_num is 47:
                    print('A 21/47 group:', group, motif_num)
                    pass
                if sorted(group) in interesting_groups:
                    print('An interesting group:', group, motif_num)

            if SAVE_COUNTED_MOTIFS:
                h = hash(frozenset(group))
                # h = frozenset(group)
                if h in self._counted_motifs:
                    print("\033[91m Group {} already counted \033[00m".format(group))
                    self._double_counter[frozenset(group)] += 1
                else:
                    self._counted_motifs.add(h)

            self._update_nodes_group(group, motif_num)
            if (i + 1) % 1000 == 0 and VERBOSE:
                self._logger.debug("Groups: %d" % i)

        # print('Max num of duplicates:', max(self._double_counter.values()))
        # print('Number of motifs counted twice:', len(self._double_counter))

        self._gnx = m_gnx


    def _get_feature(self, element):
        all_motifs = self._all_motifs.difference(set([None]))
        cur_feature = self._features[element]
        return np.array([cur_feature[motif_num] for motif_num in sorted(all_motifs)])


# consider ignoring node's data
class MotifsEdgeCalculator(MotifsNodeCalculator):
    def __init__(self, *args, include_nodes=False, **kwargs):
        self._edge_variations = {}
        self._should_include_nodes = include_nodes
        super(MotifsEdgeCalculator, self).__init__(*args, **kwargs)

    def is_relevant(self):
        # if graph is not directed, there is no use of edge variations
        return self._gnx.is_directed()

    def _calculate_motif_dictionaries(self):
        # calculating the node variations
        super(MotifsEdgeCalculator, self)._load_variations_file()
        if not self._gnx.is_directed():
            # if graph is not directed, there is no use of edge variations
            return

        motif_edges = list(permutations(range(self._level), 2))

        # level * (level - 1) is number of permutations of size 2
        num_edges = self._level * (self._level - 1)
        for group_num, motif_num in self._node_variations.items():
            bin_repr = BitArray(length=num_edges, int=group_num)
            self._edge_variations[group_num] = set([edge_type for bit, edge_type in zip(bin_repr, motif_edges) if bit])

    # noinspection PyMethodOverriding
    def _calculate(self, include=None):
        for group, group_num, motif_num in self._calculate_motif():
            if self._should_include_nodes:
                self._update_nodes_group(group, motif_num)

            for edge_type in self._edge_variations[group_num]:
                edge = tuple(map(lambda idx: group[idx], edge_type))
                if edge not in self._features:
                    self._features[edge] = {motif_number: 0 for motif_number in self._all_motifs}
                self._features[edge][motif_num] += 1


def nth_nodes_motif(motif_level):
    return partial(MotifsNodeCalculator, level=motif_level)


def nth_edges_motif(motif_level):
    return partial(MotifsNodeCalculator, level=motif_level)


feature_node_entry = {
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
}

feature_edge_entry = {
    "motif3_edge": FeatureMeta(nth_edges_motif(3), {"me3"}),
    "motif4_edge": FeatureMeta(nth_edges_motif(4), {"me4"}),
}

if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature

    # Previous version contained a bug while counting twice sub-groups with double edges
    # test_specific_feature(nth_edges_motif(3), is_max_connected=True)
    test_specific_feature(nth_edges_motif(4), is_max_connected=True)

    # def _calculate_motif_dictionaries(self):
    #     motifs_edges_dict = {}
    #     motifs_vertices_dict = {}
    #     motif_edges = list(permutations(range(self._level), 2))
    #
    #     motif_file = pandas.read_csv(self._motif_path(), delimiter="\t")
    #     if not self._gnx.is_directed():
    #         motifs_vertices_dict = {BitArray(length=3, int=int(y)).bin: int(x) for i, (x, y) in motif_file.iterrows()}
    #     else:
    #         num_edges = len(motif_edges)
    #         for _, (x, y) in motif_file.iterrows():
    #             bin_repr = BitArray(length=num_edges, int=int(y))
    #             motifs_vertices_dict[bin_repr.bin] = int(x)
    #             motifs_edges_dict[bin_repr.bin] = [edge_type for bit, edge_type in zip(bin_repr, motif_edges) if bit]
    #
    #     return {'v': motifs_vertices_dict, 'e': motifs_edges_dict}

###########################################################################################
# def _calculate(self, include=None):
#     all_motifs = set(self._node_variations.values())
#     undirected_gnx = self._gnx.to_undirected()
#     for node in self._order_by_degree():
#         history = set()
#         self._features[node] = {motif_number: 0 for motif_number in all_motifs}
#         neighbors_gnx = self._gnx.subgraph(self._get_neighborhood(node, self._level, gnx=undirected_gnx))
#         for group in self._get_subgroups(node, self._level, gnx=neighbors_gnx):
#             group = sorted(group)
#             if group in history:
#                 continue
#             history.add(group)
#             motif_number = self._get_motif_number(group)
#             self._features[node][motif_number] += 1
#         self._gnx.remove_node(node)
#
# def _subgroups(self, node, level, gnx=None):
#     if gnx is None:
#         gnx = self._gnx
#     if level == 1:
#         return node
#
# def _calculate1(self):
#     for node in self._order_by_degree():
#         history = {}
#         for sub_group in self._subgroups(node, self._level):
#             if sub_group in history:
#                 continue
#
# # this might be more efficient than dijkstra (with cutoff) - a simple BFS
# def _get_neighborhood(self, node, dist, gnx=None):
#     dist -= 1
#     if gnx is None:
#         gnx = self._gnx
#     neighborhood = set()
#     queue = [(node, 0)]
#     while queue:
#         cur_node, node_dist = queue.pop(0)
#         neighborhood.add(cur_node)
#         neighbors = set(nx.all_neighbors(gnx, cur_node)).difference(neighborhood)
#         if node_dist >= dist - 1:
#             neighborhood.update(neighbors)
#         else:  # node_dist is lower than (dist - 1)
#             queue.extend((n, node_dist + 1) for n in neighbors)
#     return neighborhood
#
# # seems more simple - but it's more costly
# def _get_neighborhood_dijkstra(self, node, dist, gnx=None):
#     if gnx is None:
#         gnx = self._gnx
#     return set(nx.single_source_dijkstra_path_length(gnx, node, cutoff=dist))
#
# def _calculate2(self):
#     self._undirected_gnx = self._gnx.to_undirected()
#     for node in self._order_by_degree(self._undirected_gnx):
#         # calculating the nth neighborhood of the node - is working on the neighborhood graph more efficient?
#         neighbors_gnx = self._gnx.subgraph(self._get_neighborhood(node, self._level))
#         history = {}
#         for sub_group in self._subgroups(node, self._level, gnx=neighbors_gnx):
#             if sub_group in history:
#                 continue
#         self._gnx.remove_node(node)


# TODO: consider removing
# def _initialize_motif_hist(self):
#     length = max(self._node_variations.values()) + 1
#     return {n: [0] * length for n in self._gnx}
#
# def _initialize_motif_hist(self):
#     node_hist = super(MotifsEdgeCalculator, self)._initialize_motif_hist()
#
#     length = max(self._edge_variations.values()) + 1
#     edge_hist = {e: [0] * length for e in self._gnx.edges()}
#     return {'v': node_hist, 'e': edge_hist}
