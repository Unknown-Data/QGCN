import pickle
from graph_features import GraphFeatures
import numpy as np
from loggers import BaseLogger, PrintLogger
import os
MOTIFS_VAR_PATH = os.path.join(__file__.rsplit(os.sep, 1)[0])


class MotifRatio:
    def __init__(self, ftr: GraphFeatures, is_directed, logger: BaseLogger=None):
        self._is_directed = is_directed                 # are the graphs directed
        self._index_ftr = None                          # list of ftr names + counter [ ... (ftr_i, 0), (ftr_i, 1) ...]
        self._logger = logger if logger else PrintLogger("graphs logger")
        # self._graph_order = graph_order if graph_order else [g for g in sorted(graph_ftr_dict)]
        self._gnx_ftr = ftr
        self._set_index_to_ftr(self._gnx_ftr)
        # list index in motif to number of edges in the motif
        self._motif_index_to_edge_num = {"motif3": self._motif_num_to_number_of_edges(3),
                                         "motif4": self._motif_num_to_number_of_edges(4)}
        self._ftr_mx = self._gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix, should_zscore=False)
        self._headers = []
        self._motif_ratio_vec = None
        self._motif_ratio_matrix = None

    # load motif variation file
    def _load_variations_file(self, level):
        fname = "%d_%sdirected.pkl" % (level, "" if self._is_directed else "un")
        fpath = os.path.join(MOTIFS_VAR_PATH, "motif_variations", fname)
        return pickle.load(open(fpath, "rb"))

    # return dictionary { motif_index: number_of_edges }
    def _motif_num_to_number_of_edges(self, level):
        motif_edge_num_dict = {}
        for bit_sec, motif_num in self._load_variations_file(level).items():
            motif_edge_num_dict[motif_num] = bin(bit_sec).count('1')
        return motif_edge_num_dict

    # map matrix columns to features + count if there's more then one from a single feature
    def _set_index_to_ftr(self, gnx_ftr):
        if not self._index_ftr:
            sorted_ftr = [f for f in sorted(gnx_ftr) if gnx_ftr[f].is_relevant()]  # fix feature order (names)
            self._index_ftr = []

            for ftr in sorted_ftr:
                len_ftr = len(gnx_ftr[ftr])
                # fill list with (ftr, counter)
                self._index_ftr += [(ftr, i) for i in range(len_ftr)]

    # get feature vector for a graph
    def _build_vector(self):
        # get gnx gnx
        final_vec = np.zeros((1, self._ftr_mx.shape[1]))

        motif3_ratio = None
        motif4_ratio = None
        for i, (ftr, ftr_count) in enumerate(self._index_ftr):
            if ftr == "motif3":
                # calculate { motif_index: motif ratio }
                motif3_ratio = self._count_subgraph_motif_by_size(self._ftr_mx, ftr) if not motif3_ratio else motif3_ratio
                final_vec[0, i] = motif3_ratio[ftr_count]
                self._headers.append("motif3_" + str(self._motif_index_to_edge_num["motif3"][ftr_count]) + "_edges")
            elif ftr == "motif4":
                # calculate { motif_index: motif ratio }
                motif4_ratio = self._count_subgraph_motif_by_size(self._ftr_mx, ftr) if not motif4_ratio else motif4_ratio
                final_vec[0, i] = motif4_ratio[ftr_count]
                self._headers.append("motif4_" + str(self._motif_index_to_edge_num["motif4"][ftr_count]) + "_edges")
            else:
                # calculate average of column
                final_vec[0, i] = np.sum(self._ftr_mx[:, i]) / self._ftr_mx.shape[0]
                self._headers.append(ftr + "_" + str(ftr_count))
        return final_vec

    def _build_matrix(self):
        sum_dictionaries_motifs3 = []
        sum_dictionaries_motifs4 = []
        # 3: [ ... row(node): { num_edges_in_motif3: count (for_this_node_only) } ... ]
        for i in range(self._ftr_mx.shape[0]):
            sum_dictionaries_motifs3.append({})
            sum_dictionaries_motifs4.append({})
            for j, (ftr, ftr_count) in enumerate(self._index_ftr):
                if ftr == "motif3":
                    key = self._motif_index_to_edge_num[ftr][ftr_count]
                    sum_dictionaries_motifs3[i][key] = sum_dictionaries_motifs3[i].get(key, 1e-3) + self._ftr_mx[i, j]
                elif ftr == "motif4":
                    key = self._motif_index_to_edge_num[ftr][ftr_count]
                    sum_dictionaries_motifs4[i][key] = sum_dictionaries_motifs4[i].get(key, 1e-3) + self._ftr_mx[i, j]

        return_mx = self._ftr_mx.copy()
        for i in range(self._ftr_mx.shape[0]):
            for j, (ftr, ftr_count) in enumerate(self._index_ftr):
                if ftr == "motif3":
                    # calculate { motif_index: motif ratio }
                    return_mx[i, j] /= sum_dictionaries_motifs3[i][self._motif_index_to_edge_num[ftr][ftr_count]]
                    if i == 0:
                        self._headers.append("motif3_" + str(self._motif_index_to_edge_num["motif3"][ftr_count]) + "_edges")
                elif ftr == "motif4":
                    # calculate { motif_index: motif ratio }
                    return_mx[i, j] /= sum_dictionaries_motifs4[i][self._motif_index_to_edge_num[ftr][ftr_count]]
                    if i == 0:
                        self._headers.append("motif4_" + str(self._motif_index_to_edge_num["motif4"][ftr_count]) + "_edges")
                else:
                    if i == 0:
                        self._headers.append(ftr + "_" + str(ftr_count))
        return return_mx

    def motif_ratio_vector(self):
        self._motif_ratio_vec = self._motif_ratio_vec if self._motif_ratio_vec else self._build_vector()
        return self._motif_ratio_vec[0]

    def motif_ratio_matrix(self):
        self._motif_ratio_matrix = self._motif_ratio_matrix if self._motif_ratio_matrix else self._build_matrix()
        return self._motif_ratio_matrix

    def get_headers(self):
        return self._headers

    # return { motif_index: sum motif in index/ total motifs with same edge count }
    def _count_subgraph_motif_by_size(self, ftr_mat, motif_type):
        sum_dict = {ftr_count: np.sum(ftr_mat[:, i]) for i, (ftr, ftr_count) in enumerate(self._index_ftr)
                    if ftr == motif_type}       # dictionary { motif_index: sum column }
        sum_by_edge = {}                        # dictionary { num_edges_in_motif: sum of  }
        for motif_count, sum_motif in sum_dict.items():
            key = self._motif_index_to_edge_num[motif_type][motif_count]
            sum_by_edge[key] = sum_by_edge.get(key, 0) + sum_motif
        # rewrite dictionary { motif_index: sum column/ total motifs with same edge count }
        for motif_count in sum_dict:
            key = self._motif_index_to_edge_num[motif_type][motif_count]
            sum_dict[motif_count] = sum_dict[motif_count] / sum_by_edge[key] if sum_by_edge[key] else 0
        return sum_dict

    # return [ ... (motif_type, counter) ... ]
    def _get_motif_type(self, motif_type, num_motifs):
        header = []
        for i in range(num_motifs):
            header.append((motif_type, i))
        return header

    @staticmethod
    def is_motif(ftr):
        return ftr == 'motif4' or ftr == "motif3"


if __name__ == "__main__":
    import networkx as nx
    from feature_meta import NODE_FEATURES

    gnx = nx.Graph()
    gnx.add_edges_from([
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 7),
        (7, 8),
        (3, 6),
        (4, 6),
        (6, 8),
        (5, 6),
    ])
    gnx_ftr = GraphFeatures(gnx, NODE_FEATURES, ".", is_max_connected=True)
    gnx_ftr.build()
    m = MotifRatio(gnx_ftr, False)
    e = 0
