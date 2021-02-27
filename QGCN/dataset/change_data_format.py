"""
If a dataset is taken from the website that has been provided, here is an example data loader in order to
adjust it to our format - a few csv files as explained in the Readme (you can adjust it to your own files and directories).
"""

def data_loader(name):
    a = np.genfromtxt("../data/{}_graph_indicator.txt".format(name), dtype=np.dtype(str))
    a = np.reshape(a, newshape=(1, a.size))
    dict_node_to_graph = {i+1: int(a[0, i]) for i in range(a.shape[1])}
    b = np.genfromtxt("../data/{}_graph_labels.txt".format(name), dtype=np.dtype(str))
    b = np.reshape(b, newshape=(1, b.size))
    dict_graph_to_label = {int(i+1): int(b[0, i]) for i in range(b.shape[1])}
    c = list(np.loadtxt("../data/{}_A.txt".format(name), dtype=np.dtype(str), delimiter=','))
    c = [tuple(list(i)) for i in c]
    dict_graph_to_edges = {k: [] for k in list(dict_graph_to_label.keys())}
    count = 0
    for e in c:
        if dict_node_to_graph[int(e[0])] == dict_node_to_graph[int(e[1])]:
            count += 1
            dict_graph_to_edges[dict_node_to_graph[int(e[0])]].append(e)

    my_list = []
    graphs = list(dict_graph_to_label.keys())
    for g in graphs:
        for e in dict_graph_to_edges[g]:
            my_dict = {"g_id": g, "src": e[0], "dst": e[1], "label": dict_graph_to_label[g]}
            my_list.append(my_dict)

    csv_columns = ["g_id", "src", "dst", "label"]
    csv_file = "{}_all.csv".format(name)
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in my_list:
                writer.writerow(data)
    except IOError:
        print("I/O error")
