import re
from collections import namedtuple
from datetime import datetime
from itertools import chain

import networkx as nx
import numpy as np

# from scipy.stats import zscore

try:
    from graph_measures.loggers import EmptyLogger
except Exception as e:
    from loggers import EmptyLogger


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
            new_matrix[:, i] = (new_matrix[:, i] - new_matrix[:, i].min()) / new_matrix[:, i].std()
    return new_matrix


def time_log(func):
    def wrapper(self, *args, **kwargs):
        start_time = datetime.now()
        self._logger.debug("Start %s" % (self._print_name,))
        res = func(self, *args, **kwargs)
        cur_time = datetime.now()
        self._logger.debug("Finish %s at %s" % (self._print_name, cur_time - start_time,))
        self._is_loaded = True  # Very bad place to put it - choose somewhere else
        return res

    return wrapper


class FeatureCalculator:
    META_VALUES = ["_gnx", "_logger"]

    def __init__(self, gnx, *args, logger=None, **kwargs):
        # super(FeatureCalculator, self).__init__()
        self._is_loaded = False
        self._features = {}
        self._logger = EmptyLogger() if logger is None else logger
        self._gnx = gnx
        self._sample_node = list(self._gnx.nodes)[0]
        self._print_name = self.print_name()
        self._default_val = 0

    is_loaded = property(lambda self: self._is_loaded, None, None, "Whether the features were calculated")

    def is_relevant(self):
        raise NotImplementedError()

    def clean_meta(self):
        meta = {}
        for name in type(self).META_VALUES:
            meta[name] = getattr(self, name)
            setattr(self, name, None)
        return meta

    def load_meta(self, meta):
        for name, val in meta.items():
            setattr(self, name, val)

    def _is_meta_loaded(self):
        return any(getattr(self, name) is not None for name in self.META_VALUES)

    @classmethod
    def print_name(cls):
        split_name = re.findall("[A-Z][^A-Z]*", cls.__name__)
        if "calculator" == split_name[-1].lower():
            split_name = split_name[:-1]
        return "_".join(map(lambda x: x.lower(), split_name))

    @time_log
    def build(self, include: set = None):
        # Don't calculate it!
        if not self.is_relevant():
            self._is_loaded = True
            return

        if include is None:
            include = set()
        self._calculate(include)
        self._is_loaded = True
        return self._features

    def _calculate(self, include):
        raise NotImplementedError()

    def _get_feature(self, element):
        raise NotImplementedError()

    def _params_order(self, input_order: list = None):
        raise NotImplementedError()

    def __len__(self):
        sample_output = self._get_feature(self._sample_node)
        try:
            sample_output = sample_output.tolist()
        except:
            pass
        return len(sample_output) if type(sample_output) == list or type(sample_output) == tuple else 1

    def to_matrix(self, params_order: list = None, mtype=np.matrix, should_zscore: bool = True):
        mx = np.matrix([self._get_feature(element) for element in self._params_order(params_order)]).astype(np.float32)
        # infinity is possible due to the change of the matrix type (i.e. overflow from 64 bit to 32 bit)
        mx[np.isinf(mx)] = self._default_val
        if 1 == mx.shape[0]:
            mx = mx.transpose()
        if should_zscore:
            mx = z_scoring(mx)  # , axis=0)
        return mtype(mx)

    def __repr__(self):
        status = "loaded" if self.is_loaded else "not loaded"
        if not self._is_meta_loaded():
            status = "no_meta"
        elif not self.is_relevant():
            status = "irrelevant"
        return "<Feature %s: %s>" % (self._print_name, status,)


# TODO: Think how to access node & edge from features. assume node/edge can be a more complicated objects

# noinspection PyAbstractClass
class NodeFeatureCalculator(FeatureCalculator):
    # def __init__(self, *args, **kwargs):
    #     super(NodeFeatureCalculator, self).__init__(*args, **kwargs)
    #     self._features = {str(node): None for node in self._gnx}

    def _params_order(self, input_order: list = None):
        if input_order is None:
            return sorted(self._gnx)
        return input_order

    def _get_feature(self, element) -> np.ndarray:
        return np.array(self._features[element])

    def edge_based_node_feature(self):
        nodes_dict = self._features
        edge_dict = {}
        for edge in self._gnx.edges():
            n1_val = np.array(nodes_dict[edge[0]])
            n2_val = np.array(nodes_dict[edge[1]])

            edge_dict[edge] = list(chain(*zip(n1_val - n2_val, np.mean([n1_val, n2_val], axis=0))))
        return edge_dict

    def feature(self, element):
        return self._get_feature(element)


# noinspection PyAbstractClass
class EdgeFeatureCalculator(FeatureCalculator):
    # def __init__(self, *args, **kwargs):
    #     super(EdgeFeatureCalculator, self).__init__(*args, **kwargs)
    #     self._features = {str(edge): None for edge in self._gnx.edges()}

    def _params_order(self, input_order: list = None):
        if input_order is None:
            return sorted(self._gnx.edges())
        return input_order

    def _get_feature(self, element) -> np.ndarray:
        return np.array(self._features[element])


FeatureMeta = namedtuple("FeatureMeta", ("calculator", "abbr_set"))
