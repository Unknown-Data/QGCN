from accelerated_features_meta import FeaturesMeta
from graph_features import GraphFeatures
import numpy as np
from motif_ratio import MotifRatio


# Old zscore code.. should use scipy.stats.zscore
def z_scoring(matrix):
    new_matrix = np.asmatrix(matrix)
    minimum = np.asarray(new_matrix.min(0))  # column wise
    for i in range(minimum.shape[1]):
        if minimum[0, i] > 0:
            new_matrix[:, i] = np.log10(new_matrix[:, i])
        elif minimum[0, i] == 0:
            new_matrix[:, i] = np.log10(new_matrix[:, i] + 0.1)
        if new_matrix[:, i].std() > 0:
            new_matrix[:, i] = (new_matrix[:, i] - new_matrix[:, i].mean()) / new_matrix[:, i].std()
    return new_matrix


def log_norm(matrix):
    matrix[np.isnan(matrix)] = 1e-10
    matrix = np.abs(matrix)
    matrix[matrix < 1e-10] = 1e-10
    return np.log(matrix)


class FeaturesProcessor:
    def __init__(self, gnx_features: GraphFeatures):
        self._gnx_ftr = gnx_features
        if not self._gnx_ftr.is_build:
            self._gnx_ftr.build()

    # function input:
    #   - to add: list/np_array/np_matrix
    #   - norm_func: function that takes a vector and outputs processed outcome
    # the function calculates a vector with ratio-motifs and average features add <to_add> and activates the
    #     <norm_func> on the vector
    # function output: the output of the above
    def activate_motif_ratio_vec(self, to_add=None, norm_func=None):
        as_vec = MotifRatio(self._gnx_ftr, self._gnx_ftr.graph.is_directed()).motif_ratio_vector()
        if to_add:
            as_vec = np.hstack((as_vec, np.matrix(to_add)))
        if norm_func:
            as_vec = norm_func(as_vec)
        return np.array(as_vec)

    @staticmethod
    def _convert_dict_to_list(dictionary, entries_order):
        if type(dictionary) == dict:
            dict_as_list = []
            for vertex in entries_order:
                dict_as_list.append(dictionary[vertex]) if type(dictionary[vertex]) == list else \
                    dict_as_list.append([dictionary[vertex]])
            return dict_as_list
        return dictionary

    # function input:
    #   - to add: list/dictionary/np_matrix
    #   - norm_func: function that takes a matrix and outputs processed outcome
    # the function adds <to_add> to the ftr_matrix and activates the <norm_func> on the matrix
    # function output: the output of the above
    def as_matrix(self, entries_order: list = None, to_add=None, norm_func=None):
        entries_order = entries_order if entries_order else sorted(self._gnx_ftr.graph)
        as_matrix = self._gnx_ftr.to_matrix(entries_order=entries_order, dtype=np.float32, mtype=np.matrix,
                                            should_zscore=False)
        if to_add:
            to_add = self._convert_dict_to_list(to_add, entries_order)
            as_matrix = np.hstack((as_matrix, np.matrix(to_add)))
        if norm_func:
            as_matrix = norm_func(as_matrix)
        return as_matrix


if __name__ == "__main__":
    import networkx as nx

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
    gnx_ftr = GraphFeatures(gnx, FeaturesMeta().NODE_LEVEL, ".", is_max_connected=True)
    fp = FeaturesProcessor(gnx_ftr)
    fp.activate_motif_ratio_vec()
    e = 0
