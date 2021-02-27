import os
import sys
import networkx as nx
import pickle

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

from src.accelerated_graph_features.feature_wrappers import motif

FEATURES = {
    'Motif3': lambda g: motif(g, level=3,gpu=False),
    'Motif4': lambda g: motif(g, level=4,gpu=False)
}


def calculate_features_on_graph(file_name, feats):
    save_file_name = file_name + '_feature_results_{}.pickle'

    g = nx.read_gpickle(file_name)

    for k, func in feats.items():
        print(k)
        res = func(g)
        with open(save_file_name.format(k), 'w+b') as fp:
            pickle.dump(res, fp)
        del res


if __name__ == '__main__':
    calculate_features_on_graph("itay_n_1000_p_0.5_size_20_160119", FEATURES)
