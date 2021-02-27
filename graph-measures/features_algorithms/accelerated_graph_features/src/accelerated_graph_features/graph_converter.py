import networkx as nx


def convert_graph_to_db_format(input_graph: nx.Graph, with_weights=False, cast_to_directed=False):
    """Converts a given graph into a DB format, which consists of two or three lists


        1. **Index list:** a list where the i-th position contains the index of the beginning of the list of adjacent nodes (in the second list).
        2. **Node list:** for each node, we list (in order) all the nodes which are adjacent to it.
        3. **Weight list:** if the weight parameter is True, includes the weights of the edges, corresponds to the nodes list

    **Assumptions:**

    The code has several preexisting assumptions:

        a) The nodes are labeled with numbers
        b) Those numbers are the sequence [0,...,num_of_nodes-1]
        c) If there are weights, they are floats
        d) If there are weights, they are initialized for all edges
        e) If there are weights, the weight key is 'weight'

    .. Note::
        The code behaves differently for directed and undirected graphs.
	For undirected graph, every edge is actually counted twice (p->q and q->p).

    Example::
		For the simple directed graph (0->1, 0->2,0->3,2->0,3->1,3->2):

		`Indices: [0, 3, 3, 4, 6]`
		`Neighbors: [1, 2, 3, 0, 1, 2]`

        Note that index[1] is the same as index[2]. That is because 1 has no neighbors, and so his neighbor list is of size 0, but we still need to have an index for the node on.

    For the same graph when it is undirected:
        `Indices: [0, 3, 5, 7, 10]`
        `Neighbors: [1, 2, 3, 0, 3, 0, 3, 0, 1, 2]`

        Note that the number of edges isn't doubled because in the directed version there is a bidirectional edge.



    :param graph: the nx.Graph object to convert
    :param with_weights: whether to create a weight list. Defaults to False.
    :param cast_to_directed: whether to cast the graph into a directed format

    :return: two or three lists: index,nodes, [weights]
    """

    if cast_to_directed:
        graph = input_graph.to_directed()
    else:
        graph = input_graph.copy()

    if graph.is_directed():
        # Color printing taken from https://www.geeksforgeeks.org/print-colors-python-terminal/
        print("\033[93m {}\033[00m".format('Note that the graph is processed as a directed graph'))

    indices = [0]  # The first neighbor list always starts at index 0
    neighbor_nodes = []

    nodes = [node for node in graph.nodes()]
    # print(nodes)
    nodes.sort()
    neighbors = [sorted([x for x in graph.neighbors(node)]) for node in nodes]

    # Create the indices and neighbor nodes lists
    for neighbor_list in neighbors:
        neighbor_list.sort()
        # print(neighbor_list)
        neighbor_nodes.extend(neighbor_list)
        indices.append(indices[-1] + len(neighbor_list))

    if with_weights:
        try:
            weights = [0] * len(neighbor_nodes)
            current_index = 0
            for node in nodes:
                for x in neighbors[node]:
                    w = graph[node][x]['weight']
                    weights[current_index] = w
                    current_index += 1

            return indices, neighbor_nodes, weights
        except KeyError:
            # Print in red
            print("\033[91m {}\033[00m".format('No weights defined, returning an empty list of weights'))
            print()
            return indices, neighbor_nodes, []

    return indices, neighbor_nodes


def convert_graph_to_db_dict(graph: nx.Graph, with_weights=False, cast_to_directed=False):
    """
    Encapsulates the convert_graph_to_db_format function by wrapping the returned lists with a dictionary.
    The dictionary has the keys 'indices','neighbors' , ['weights'].

    :param graph: the nx graph to convert
    :param with_weights: whether to return a weights list
    :param cast_to_directed: whether to process the graph as a directed one.
    :return: a dictionary with the specified keys and the lists as values.
    """
    ret_dict = {}
    if with_weights:
        i, n, w = convert_graph_to_db_format(graph, with_weights, cast_to_directed)
        ret_dict['indices'] = i
        ret_dict['neighbors'] = n
        ret_dict['weights'] = w
        ret_dict['with_weights'] = True
    else:
        i, n = convert_graph_to_db_format(graph, with_weights, cast_to_directed)
        ret_dict['indices'] = i
        ret_dict['neighbors'] = n
        ret_dict['with_weights'] = False

    ret_dict['directed'] = graph.is_directed()
    return ret_dict
