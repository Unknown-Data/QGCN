import networkx as nx

from graph_infra.union_graph import UnionGraph, GraphNode


class AttrMultiDiGraph(UnionGraph):
    def __init__(self, data=None, **attr):
        attr["attr_names"] = set()
        super(AttrMultiDiGraph, self).__init__(data, **attr)

    def add_edge(self, u, v, key=None, attr_dict=None, **attr):
        super(AttrMultiDiGraph, self).add_edge(u, v, key, attr_dict, **attr)
        attribute = attr.get("attr_name", None) if attr_dict is None else attr_dict.get("attr_name", None)
        self.graph["attr_names"].add(attribute)

    @classmethod
    def _join_in_edges(cls, edges, nodes, new_node, self_loop=False):
        joined_edges = cls._join_out_edges([(n2, n1, data) for n1, n2, data in edges], nodes, new_node, self_loop)
        return [(n2, n1, data) for n1, n2, data in joined_edges]

    # Joining out edges
    # @param: nodes(set) all nodes to be joined
    # @param: edges(list/iterable) all out_edges from any node in the nodes' set
    @classmethod
    def _join_out_edges(cls, edges, nodes, new_node, self_loop=False):
        # edges: list, nodes: set
        # set(edges, key=lambda x: x[l]) <-> to implement this + Change edges[0] with new_node
        # data = {"ff": #num} data1 == data2 || data1["ff"] == data2["ff"]
        unique_edges = set([(new_node if n2 in nodes else n2, tuple(data.items())) for n2, data in edges])
        return [(new_node, n2, dict(data)) for n2, data in unique_edges if (n2 != new_node) or self_loop]

    def merge_nodes(self, nodes, self_loop=False):
        if not nodes:
            return
        nodes = sorted(nodes, key=lambda n: n.timestamp)
        in_edges = list(self.in_edges(nodes, data=True))
        out_edges = list(self.out_edges(nodes, data=True))
        self.remove_edges_from(map(lambda x: (x[0], x[1]), in_edges + out_edges))
        self.remove_nodes_from(nodes)

        # Joining nodes
        # base_node = reduce(lambda n1, n2: n1.join(n2), nodes)  # Not compatible with python3
        joined_node = nodes[0]
        for node in nodes[1:]:
            joined_node = joined_node.join(GraphNode.real_node(node))

        # Joining edges
        new_in_edges = self._join_in_edges(in_edges, nodes, joined_node, self_loop)
        new_out_edges = self._join_out_edges(out_edges, nodes, joined_node, self_loop)

        self.add_edges_from(new_in_edges)
        self.add_edges_from(new_out_edges)

    def subgrapn_attr(self, data):
        h = nx.DiGraph(data=data)
        h.add_nodes_from(set(map(lambda n: GraphNode(n.node_id), self)))
        h.add_edges_from(map(lambda e: (e[0], e[1]), self.edges(data=data)))
        return h

    # def get_nodes(self, node, dist_m, time_gap):
    #     relevant_nodes = filter(lambda n: node.timestamp < n.timestamp < node.timestamp + time_gap, self.nodes())
    #     for node2 in sorted(relevant_nodes, key=lambda n: n.timestamp):
    #         pass
    #     return []
    #
    # def cluster_subgraph(self, dist_m, time_gap, self_loop=False):
    #     for node in sorted(self, key=lambda x: x.timestamp):
    #         self.merge_nodes(self.get_nodes(node, dist_m, time_gap), self_loop=self_loop)
    #
    # def cluster_graph(self, dist_m, time_gap, self_loop=False):
    #     for attr in self.data_attrs():
    #         subgraph = self.subgraph_attr(attr)
    #         subgraph.cluster_subgraph(dist_m, time_gap, self_loop=self_loop)


def test_union_graph():
    g = AttrMultiDiGraph()
    g.add_edges_from([
        (GraphNode(1), GraphNode(2), {"f": 1, "k": 2}),
        (GraphNode(1), GraphNode(3), {"r": 1, "f": 3}),
        (GraphNode(1), GraphNode(4), {"f": 1, "k": 2}),
        (GraphNode(2), GraphNode(4), {"r": 2, "f": 4}),
        (GraphNode(2), GraphNode(3), {"r": 3, "f": 4})
    ])
    g.add_edge(GraphNode(5), GraphNode(6), f=9)
    # e = g.edges()
    print("Bla")


def test_union_node():
    ns = set(GraphNode(x, data=(5 * x, 5 * x + 2), timestamp=8 * x) for x in range(8))
    print("bla")
    # node:
    #   time
    #   data


if __name__ == "__main__":
    test_union_graph()
