from src.accelerated_graph_features.graph_converter import convert_graph_to_db_dict
import networkx as nx

class FeatureWrapper(object):
    """
    This class is a decorator for the pure python function that are exposed to the user of the accelerated feature package.
    The decorator is responsible for doing the tasks that are common to all wrapper functions:
         - Converting the nx.Graph object to a converted graph dictionary.
         - Marking the time for conversion and calculation (if timer is given)
    """

    def __init__(self, func):
        self.f = func

    def _convert_gnx_to_int_based(self, gnx: nx.Graph):
        nodes = {node: i for i, node in enumerate(gnx.nodes)}
        new_gnx = nx.DiGraph() if gnx.is_directed() else nx.Graph()
        for u, v in gnx.edges:
            new_gnx.add_edge(nodes[u], nodes[v])
        return new_gnx, [n for n, i in sorted(nodes.items(), key=lambda x: x[1])]

    def __call__(self, graph, **kwargs):
        with_weights = kwargs.get('with_weights', False)
        cast_to_directed = kwargs.get('cast_to_directed', False)

        new_graph, node_map = self._convert_gnx_to_int_based(graph)
        converted_graph = convert_graph_to_db_dict(new_graph, with_weights, cast_to_directed)

        if 'timer' in kwargs:
            kwargs['timer'].mark()

        res = self.f(converted_graph, **kwargs)

        if 'timer' in kwargs:
            kwargs['timer'].stop()

        return {name: val for name, val in zip(node_map, res)}
