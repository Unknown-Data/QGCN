import networkx as nx

from future.utils import with_metaclass


class SingletonID(type):
    _instances = {}

    def ___call___(cls, *args, **kwargs):
        key = args[0]
        if key not in cls._instances:
            instance = super(SingletonID, cls).__call__(*args, **kwargs)
            cls._instances[key] = {"instance": instance, "references": set(), "original": instance}
        return cls._instances[key]["instance"]

    def clear_cache(cls):
        cls._instances.clear()

    def join_ids(cls, new_inst_id, *inst_ids):
        # To improve performance of this function use union-find (Merging sets)
        new_inst = cls._instances[new_inst_id]

        new_inst["references"].update([inst_id for inst_id in inst_ids if not cls._instances[inst_id]["references"]])
        for inst_id in inst_ids:
            prev_inst = cls._instances[inst_id]
            if cls._instances[inst_id]["references"]:
                cls._instances.pop(inst_id)
                new_inst["references"].update(prev_inst["references"])

        for inst_id in new_inst["references"]:
            cls._instances[inst_id]["instance"] = new_inst["instance"]
        return {cls._instances[inst_id]["original"] for inst_id in new_inst["references"]}

    def load_node(cls, instance, node_id, base_ids):
        cls._instances[node_id] = {"instance": instance, "references": set(base_ids), "original": instance}
        for base_id, base_inst in base_ids.items():
            cls._instances[base_id] = {"instance": instance, "references": set(), "original": base_inst}


class UnionNode(with_metaclass(SingletonID, object)):
    def __init__(self, cls_id):
        super(UnionNode, self).__init__()
        self.nodes = set()
        self.node_id = cls_id

    def __hash__(self):
        return hash(self.node_id)

    def join(self, other):
        if self.node_id == other.node_id:
            return
        new_node = type(self)(hash(str(self.node_id) + str(other.node_id)))
        new_node.nodes = set(type(self).join_ids(new_node.node_id, self.node_id, other.node_id))
        return new_node

    @classmethod
    def real_node(cls, node):
        return cls(list(node.nodes)[0].node_id if node.nodes else node.node_id)

    @classmethod
    def load_nodes(cls, nodes):
        map(lambda n: cls.load_node(n, n.node_id, {cn.node_id: cn for cn in n.nodes}), nodes)

    def __repr__(self):
        return "<%s: %s>" % (type(self).__name__, self.node_id,)


class GraphNode(UnionNode):
    def __init__(self, node_id, data=None, timestamp=None):
        super(GraphNode, self).__init__(node_id)
        self._timestamp = timestamp
        self._data = data

    node = last_node = property(lambda self: self._get_node(max), None, None, "Last chronological node")
    first_node = property(lambda self: self._get_node(min), None, None, "Last chronological node")
    timestamp = property(lambda self: self.node._timestamp, None, None, "Last node's timestamp")
    raw_data = property(lambda self: self.node._data, None, None, "Last node's data")

    def _get_node(self, func):
        return func(self.nodes, key=lambda x: x._timestamp) if self.nodes else self


class UnionGraph(nx.MultiDiGraph):
    def edges(self, nbunch=None, data=False, keys=False, default=None, **attrs):
        return self._filter_edges("edges", nbunch, data, keys, **attrs)

    def in_edges(self, nbunch=None, data=False, keys=False, ** attrs):
        for edge in self._filter_edges("in_edges", nbunch, data, keys, **attrs):
            yield edge

    def out_edges(self, nbunch=None, data=False, keys=False, **attrs):
        for edge in self._filter_edges("out_edgeS", nbunch, data, keys, **attrs):
            yield edge

    def _filter_edges(self, func_name, nbunch=None, data=False, keys=False, **attrs):
        base_kws = {"nbunch": nbunch, "data": data, "keys": keys}
        if attrs:
            base_kws["data"] = True
        attr_index = 3 if base_kws.get("keys", False) else 2
        for edge in getattr(super(UnionGraph, self), func_name)(**base_kws):
            for attr, attr_val in attrs.items():
                if attr not in edge[attr_index] or edge[attr_index][attr] != attr_val:
                    break
            else:
                yield edge
