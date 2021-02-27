"""
    all the graphs are stored under one graph
    - each edge has a set of attributes - {graph_1: weight_1, ...... graph_k: weight_k}
    - if the database files name are with dates, use date_format='format' for example date_format='%d-%b-%Y'
"""
from loggers import PrintLogger, BaseLogger
import networkx as nx
import os


class MultiGraph:
    """
    database_name:  name of the data set
    source=

        None: in that case, it is expected that a dir named DATABASE_NAME will be in /GRAPHS_INPUT/multi_graph
              the directory should contain files, each file is a single graph, in the following format:
              each line is a single edge    source,dest,weight           # weight is optional
        PATH: same as above but the directory will be the given path
        DICTIONARY { name: gnx }
        DICTIONARY { name: edge_list=[...(source,dest,weight)...] }      # weight is optional
    """
    def __init__(self, database_name, graphs_source=None, directed=False, logger: BaseLogger=None):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._databese_name = database_name
        self._directed = directed
        self._gnx_multi = nx.DiGraph() if directed else nx.Graph()
        self._logger = logger if logger else PrintLogger("multi-graph")
        self._source = graphs_source

        self._list_id = []          # list of graph ID's - gives a mapping index to ID
        self._dict_id = {}          # dictionary of graph ID's - gives a mapping ID to index
        self._node_count = {}
        self._node_lists = {}
        self._edge_count = {}
        self._graph_valid = {}

        self._build(self._source)
        self._logger.debug("finish initialization")

    def suspend_logger(self):
        self._logger = None

    def wake_logger(self, logger: BaseLogger=None):
        self._logger = logger if logger else PrintLogger("multi-graph")

    def _build(self, source):
        if not source:
            source = os.path.join(self._base_dir, "GRAPHS_INPUT", "multi_graph", self._databese_name)
        if type(source) == str:
            self._build_from_files(source)
        elif type(source) == dict and len(source) > 0:
            item = next(iter(source.values()))
            if type(item) == nx.classes.graph.Graph or type(item) == nx.classes.digraph.DiGraph:
                self._build_from_gnx_dict(source)
            if type(item) == list and len(item) > 0 and type(item[0]) == tuple:
                self._build_from_edge_list_dict(source)
        else:
            self._logger.error("graph_source is not in the correct format")
            exit(1)

    @staticmethod
    def _strip_txt(file_name):
        return file_name.split(".")[0]

    @staticmethod
    def _split_row(row):
        row = row.split()
        edge = [1, 1, 1]
        for i, val in enumerate(row):
            edge[i] = val
        return edge

    def _add_edge_to_gnx(self, graph_name, node_u, node_v, weight):
        self._logger.debug("adding edge:\t(" + str(node_u) + "," + str(node_v) + ")\tweight=" + str(weight))

        # count edges
        curr_edge_count = self._edge_count.get(graph_name, 0)
        # counter++ if:
        #   1. the edge exists in the multi_graph but it doesnt belong to the this specific graph
        #   2. the edge is not in the multi graph at all
        if (self._gnx_multi.has_edge(node_u, node_v) and graph_name not in self._gnx_multi.edges[node_u, node_v]) or \
                not self._gnx_multi.has_edge(node_u, node_v):
            self._edge_count[graph_name] = curr_edge_count + 1

        # count nodes
        curr_node_count = self._node_count.get(graph_name, 0)
        curr_node_list = self._node_lists.get(graph_name, [])
        if node_u not in curr_node_list:
            curr_node_count += 1
            curr_node_list += [node_u]
        if node_v not in curr_node_list:
            curr_node_count += 1
            curr_node_list += [node_v]
        self._node_count[graph_name] = curr_node_count
        self._node_lists[graph_name] = curr_node_list

        # add the edge to the graph if it doesn't exist
        self._gnx_multi.add_edge(node_u, node_v)
        # add {graph_name=weight_under_the_graph} to the attributes of the edge
        self._gnx_multi.edges[node_u, node_v][graph_name] = float(weight)

    def _index_graph(self, graph_name, valid=True):
        # save graph indexing
        if graph_name not in self._dict_id:
            self._dict_id[graph_name] = len(self._list_id)
            self._list_id.append(graph_name)
            self._graph_valid[graph_name] = valid

    def _build_from_files(self, source):
        # check source file exist
        if not source or not os.path.exists(source):
            self._logger.error("\n\n\t\t*** THE PATH TO THE DATABASE FILES IS NOT VALID !!! ***\n")
            exit(1)

        for graph_name in sorted(os.listdir(source)):
            graph_file = open(os.path.join(source, graph_name))
            graph_name = self._strip_txt(graph_name)
            # save graph indexing
            self._index_graph(graph_name)
            for row in graph_file:
                [node_u, node_v, weight] = self._split_row(row)
                self._add_edge_to_gnx(graph_name, node_u, node_v, weight)

            graph_file.close()

    def _build_from_edge_list_dict(self, source):
        for graph_name, edge_list in source.items():
            # save graph indexing
            self._index_graph(graph_name)
            for edge in edge_list:
                node_u = edge[0]
                node_v = edge[1]
                weight = 1 if len(edge) < 3 else edge[2]
                self._add_edge_to_gnx(graph_name, node_u, node_v, weight)

    def _build_from_gnx_dict(self, source):
        for graph_name, gnx in source.items():
            # save graph indexing
            self._index_graph(graph_name)
            for edge in gnx.edges(data=True):
                node_u = edge[0]
                node_v = edge[1]
                weight = edge[2]['weight'] if 'weight' in edge[2] else 1
                self._add_edge_to_gnx(graph_name, node_u, node_v, weight)

    def _subgraph_by_name(self, graph_name: str):
        if not self.is_graph(graph_name):
            self._logger.error("no graph named:\t" + graph_name)
            return
        subgraph_edges = nx.DiGraph() if self._directed else nx.Graph()
        for edge in list(self._gnx_multi.edges(data=True)):
            if graph_name in edge[2]:
                # edge is saved in the following method (from, to, {graph_name_i: weight_in_graph_i})
                subgraph_edges.add_edge(edge[0], edge[1], weight=float(edge[2][graph_name]))
        return subgraph_edges

    def _subgraph_by_index(self, index: int):
        if index < 0 or index > len(self._list_id):
            self._logger.error("index is out of scope - index < number_of_graphs/ index > 0")
        return self._subgraph_by_name(self._list_id[index])

    @staticmethod
    def _graph_from_list_in_dict(names_list, names_dict):
        for name in names_list:
            if name in names_dict:
                return True
        return False

    def _combined_graph_by_names(self, names_list=None, combine_all=False):
        # case combine_all - the returned graph will combination of all graphs regardless to the names list
        if combine_all:
            names_list = self._list_id
        else:
            for graph_name in names_list:
                if not self.is_graph(graph_name):
                    self._logger.error("no graph named:\t" + graph_name)
                    return
        subgraph_edges = nx.DiGraph() if self._directed else nx.Graph()
        for edge in list(self._gnx_multi.edges(data=True)):
            if self._graph_from_list_in_dict(names_list, edge[2]):
                # edge is saved in the following method (from, to, {graph_name_i: weight_in_graph_i})
                subgraph_edges.add_edge(edge[0], edge[1])
        return subgraph_edges

    def _combined_graph_by_indexes(self, index_list=None, combine_all=False):
        for i, index in enumerate(index_list):
            if index < 0 or index > len(self._list_id):
                self._logger.error("index is out of scope - index < number_of_graphs/ index > 0")
                return
            index_list[i] = self._list_id[index]
        return self._combined_graph_by_names(index_list, combine_all)

    def add_edges(self, source):
        if source:
            self._build(source)

    def remove_edges(self, edge_lists: dict):
        self._node_lists = {}
        self._node_count = {}
        self._graph_valid = {}
        self._list_id = []
        self._dict_id = {}

        # iterate over all edges
        for edge in list(self._gnx_multi.edges(data=True)):
            # if the edge needs to be removed for some specific graph remove the graph name from edge dictionary
            for graph_name in list(edge[2].keys()):
                self._index_graph(graph_name)
                if graph_name in edge_lists and \
                        ((edge[0], edge[1]) in edge_lists[graph_name] or
                         (not self._directed and (edge[1], edge[0]) in edge_lists[graph_name])):
                    del edge[2][graph_name]
                    self._edge_count[graph_name] -= 1
                else:
                    # count nodes
                    curr_node_count = self._node_count.get(graph_name, 0)
                    curr_node_list = self._node_lists.get(graph_name, [])
                    if edge[0] not in curr_node_list:
                        self._node_count[graph_name] = curr_node_count + 1
                        self._node_lists[graph_name] = curr_node_list + [edge[0]]
                    if edge[1] not in curr_node_list:
                        self._node_count[graph_name] += 1
                        self._node_lists[graph_name] = curr_node_list + [edge[1]]

            if len(edge[2]) == 0:
                self._gnx_multi.remove_edge(edge[0], edge[1])
        self._gnx_multi.remove_nodes_from(list(nx.isolates(self._gnx_multi)))

    def filter(self, func, func_input='gnx'):
        self.wake_logger()
        for i, gid in enumerate(self._list_id):
            if func_input == 'gnx':
                self._graph_valid[gid] = func(self._subgraph_by_name(gid))
            if func_input == 'graph_name':
                self._graph_valid[gid] = func(gid)
            if func_input == 'index':
                self._graph_valid[gid] = func(i)
            self._logger.info("filter graph " + gid + " :" + str(self._graph_valid[gid]))

    def clear_filters(self):
        for i in self._list_id:
            self._graph_valid[i] = True

    def sort_by(self, key_func):
        new_order = [i for i in sorted(self._list_id, key=key_func)]
        new_dict_id_to_index = {}
        for i, graph_id in enumerate(new_order):
            new_dict_id_to_index[graph_id] = i
        self._list_id = new_order
        self._dict_id = new_dict_id_to_index

    def get_gnx(self, graph_id):
        return self._subgraph_by_index(graph_id) if type(graph_id) == int else self._subgraph_by_name(graph_id)

    def combined_gnx(self, id_list: list=None):
        # return all graphs together
        if not list:
            return self._combined_graph_by_names(combine_all=True)
        # return by list
        if len(id_list) == 0:
            return nx.DiGraph() if self._directed else nx.Graph()
        return self._combined_graph_by_names(names_list=id_list) if type(id_list[0]) == str \
            else self._combined_graph_by_indexes(index_list=id_list)

    def is_graph(self, graph_id):
        if type(graph_id) == str:
            return True if graph_id in self._dict_id else False
        return True if graph_id < self.number_of_graphs() else False

    def index_to_name(self, index):
        if index < 0 or index > len(self._list_id):
            self._logger.error("index is out of scope - index < number_of_graphs/ index > 0")
            return None
        return self._list_id[index]

    def name_to_index(self, graph_name):
        if not self.is_graph(graph_name):
            self._logger.error("no graph named:\t" + graph_name)
            return None
        return self._dict_id[graph_name]

    def node_count(self, graph_id=None):
        # return count for specific graph
        if graph_id is not None:
            return self._node_count[self.index_to_name(graph_id)] if type(graph_id) is int else self._node_count[graph_id]
        # return list of edge count for all graphs
        return [self._node_count[i] for i in self._list_id]

    def edge_count(self, graph_id=None):
        # return count for specific graph
        if graph_id is not None:
            return self._edge_count[self.index_to_name(graph_id)] if type(graph_id) is int else self._edge_count[graph_id]
        # return list of edge count for all graphs
        return [self._edge_count[i] for i in self._list_id]

    def graphs(self, start_id=None, end_id=None):
        for gid in self._list_id[start_id: end_id]:
            if self._graph_valid[gid]:
                yield self._subgraph_by_name(gid)

    def graph_names(self, start_id=None, end_id=None):
        for gid in self._list_id[start_id: end_id]:
            if self._graph_valid[gid]:
                yield gid

    def number_of_graphs(self):
        return len(self._list_id)


def multi_graph_testing():
    mg = MultiGraph("EnronInc_by_day")
    mg.filter(lambda x: False if x.number_of_nodes() < 1500 else True)
    e = 0


if __name__ == "__main__":
    # pass
    multi_graph_testing()
