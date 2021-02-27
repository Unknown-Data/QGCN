import os
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/accelerated_graph_features'))
from functools import partial

import numpy as np
from features_algorithms.accelerated_graph_features.src import motif
from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta

CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(os.path.dirname(CUR_PATH))


class MotifsNodeCalculator(NodeFeatureCalculator):
    def __init__(self, *args, level=3, gpu=False, **kwargs):
        super(MotifsNodeCalculator, self).__init__(*args, **kwargs)
        assert level in [3, 4], "Unsupported motif level %d" % (level,)
        self._level = level
        self._gpu = gpu
        self._print_name += "_%d" % (self._level,)

    def is_relevant(self):
        return True

    @classmethod
    def print_name(cls, level=None):
        print_name = super(MotifsNodeCalculator, cls).print_name()
        if level is None:
            return print_name
        return "%s_%d_C_kernel" % (print_name, level)

    def _calculate(self, include=None):
        self._features = motif(self._gnx, level=self._level, gpu=self._gpu)

    def _get_feature(self, element):
        return np.array(self._features[element])


def nth_nodes_motif(motif_level, gpu):
    return partial(MotifsNodeCalculator, level=motif_level, gpu=gpu)


feature_node_entry = {
    "motif3_c": FeatureMeta(nth_nodes_motif(3, False), {"m3_c"}),
    "motif4_c": FeatureMeta(nth_nodes_motif(4, False), {"m4_c"}),
}
