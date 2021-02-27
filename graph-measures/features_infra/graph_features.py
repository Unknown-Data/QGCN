import os
import pickle

from multiprocessing import Process, Queue

from features_infra.feature_calculators import FeatureCalculator
from loggers import PrintLogger, EmptyLogger

import networkx as nx
import numpy as np
from scipy import sparse
from operator import itemgetter as at


class Worker(Process):
    def __init__(self, queue, calculators, include, logger=None):
        super(Worker, self).__init__()
        if logger is None:
            logger = EmptyLogger()

        self._queue = queue
        self._calculators = calculators
        self._logger = logger
        self._include = include

    def run(self):
        self._logger.info('Worker started')
        # do some initialization here

        self._logger.info('Computing things!')
        for feature_name in iter(self._queue.get, None):
            self._calculators[feature_name].build(include=self._include)


# object that calculates & holds a list of features of a graph.
class GraphFeatures(dict):
    def __init__(self, gnx, features, dir_path, logger=None, is_max_connected=False):
        self.is_build = False
        self._base_dir = dir_path
        self._logger = EmptyLogger() if logger is None else logger
        self._matrix = None

        if is_max_connected:
            if gnx.is_directed():
                subgraphs = nx.weakly_connected_component_subgraphs(gnx)
            else:
                subgraphs = nx.connected_component_subgraphs(gnx)
            self._gnx = max(subgraphs, key=len)
        else:
            self._gnx = gnx

        self._abbreviations = {abbr: name for name, meta in features.items() for abbr in meta.abbr_set}

        # building the feature calculators data structure
        super(GraphFeatures, self).__init__({name: meta.calculator(self._gnx, logger=logger)
                                             for name, meta in features.items()})

    @property
    def graph(self):
        return self._gnx

    def _build_serially(self, include, force_build: bool = False, dump_path: str = None):
        if dump_path is not None and self._gnx is not None:
            pickle.dump(self._gnx, open(self._feature_path("gnx", dump_path), "wb"))
        for name, feature in self.items():
            if force_build or not os.path.exists(self._feature_path(name)):
                feature.build(include=include)
                if dump_path is not None:
                    self._dump_feature(name, feature, dump_path)
            else:
                self._load_feature(name)

    # a single process means it is calculated serially
    def build(self, num_processes: int = 1, include: set = None, should_dump: bool = False, force_build=False):  # , exclude: set=None):
        # if exclude is None:
        #     exclude = set()
        if include is None:
            include = set()

        if 1 == num_processes:
            dump_path = None
            if should_dump:
                dump_path = self._base_dir
                if not os.path.exists(dump_path):
                    os.makedirs(dump_path)
            return self._build_serially(include, dump_path=dump_path, force_build=force_build)

        request_queue = Queue()
        workers = [Worker(request_queue, self, include, logger=self._logger) for _ in range(num_processes)]
        # Starting all workers
        for worker in workers:
            worker.start()

        # Feeding the queue with all the features
        for feature_name in self:
            request_queue.put(feature_name)

        # Sentinel objects to allow clean shutdown: 1 per worker.
        for _ in range(num_processes):
            request_queue.put(None)

        # Joining all workers
        for worker in workers:
            worker.join()

        self.is_build = True

    def _load_feature(self, name):
        if self._gnx is None:
            assert os.path.exists(self._feature_path("gnx")), "Graph is not present in the given directory"
            self._gnx = pickle.load(open(self._feature_path("gnx"), "rb"))
        feature = pickle.load(open(self._feature_path(name), "rb"))
        feature.load_meta({name: getattr(self, name) for name in FeatureCalculator.META_VALUES})
        self[name] = feature
        return self[name]

    def __getattr__(self, name):
        if name not in self:
            if name in self._abbreviations:
                name = self._abbreviations[name]
            else:
                return super(GraphFeatures, self).__getattribute__(name)

        # if obj is already calculated - return it
        obj = self[name]
        if obj.is_loaded:
            return obj

        # if obj is not calculated, check if it exist on the file system
        # if it doesn't - calculate it, if it does - load it and return it
        if not os.path.exists(self._feature_path(name)):
            obj.build()
            return obj

        return self._load_feature(name)

    @property
    def features(self):
        return set(self)

    def _feature_path(self, name, dir_path=None):
        if dir_path is None:
            dir_path = self._base_dir
        return os.path.join(dir_path, name + ".pkl")

    def _dump_feature(self, name, feature, dir_path):
        if feature.is_loaded:
            prev_meta = feature.clean_meta()  # in order not to save unnecessary data
            pickle.dump(feature, open(self._feature_path(name, dir_path), "wb"))
            feature.load_meta(prev_meta)

    def dump(self, dir_path=None):
        if dir_path is None:
            dir_path = self._base_dir

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for name, feature in self.items():
            self._dump_feature(name, feature, dir_path)

    # sparse.csr_matrix(matrix, dtype=np.float32)
    def to_matrix(self, entries_order: list = None, add_ones=False, dtype=None, mtype=np.matrix,
                  should_zscore: bool = True):
        if entries_order is None:
            entries_order = sorted(self._gnx)

        sorted_features = map(at(1), sorted(self.items(), key=at(0)))
        # Consider caching the matrix creation (if it takes long time)
        sorted_features = [feature for feature in sorted_features if feature.is_relevant() and feature.is_loaded]

        if sorted_features:
            # / added to fix dimensions problem
            if len(entries_order) == 1:
                mx = np.hstack([feature.to_matrix(entries_order, mtype=mtype, should_zscore=should_zscore).transpose()
                                for feature in sorted_features])
            # added /
            else:
                mx = np.hstack([feature.to_matrix(entries_order, mtype=mtype, should_zscore=should_zscore)
                                for feature in sorted_features])
            if add_ones:
                mx = np.hstack([mx, np.ones((mx.shape[0], 1))])
            mx.astype(dtype)
        else:
            mx = np.matrix([])

        return mtype(mx)

    def to_dict(self, dtype=None, should_zscore: bool = True):
        mx = self.to_matrix(dtype=dtype, mtype=np.matrix, should_zscore=should_zscore)
        return {node: mx[i, :] for i, node in enumerate(sorted(self._gnx))}


# class GraphNodeFeatures(GraphFeatures):
#     def sparse_matrix(self, entries_order: list=None, **kwargs):
#         if entries_order is None:
#             entries_order = sorted(self._gnx)
#         return super(GraphNodeFeatures, self).sparse_matrix(entries_order=entries_order, **kwargs)
#
#
# class GraphEdgeFeatures(GraphFeatures):
#     def sparse_matrix(self, entries_order: list=None, **kwargs):
#         if entries_order is None:
#             entries_order = sorted(self._gnx.edges())
#         return super(GraphEdgeFeatures, self).sparse_matrix(entries_order=entries_order, **kwargs)


if __name__ == "__main__":
    from feature_meta import ALL_FEATURES

    ftrs = GraphFeatures(nx.DiGraph(), ALL_FEATURES)
    print("Bla")
