import sys
import os

# Leave the path changes here!!!
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'graph_measures'))


from src.accelerated_graph_features.feature_wrappers import clustering_coefficient, k_core, node_page_rank, bfs_moments, \
    motif,attraction_basin,flow
from src.accelerated_graph_features.graph_timer import FeatureTimer
from src.accelerated_graph_features.original_features_check import original_bfs_moments
# from Git.graph_measures.cpp_comparison import compare_motifs
from graph_measures.features_algorithms.vertices.attractor_basin import AttractorBasinCalculator
from graph_measures.features_algorithms.vertices.flow import FlowCalculator
import networkx as nx

matching_python_functions = {
    # 'clustering': [clustering_coefficient, nx.average_clustering],
    # 'k_core': [k_core, nx.k_core],
    # 'page_rank': [node_page_rank, nx.pagerank]
    # 'BFSMoments': [bfs_moments, original_bfs_moments]
    # 'Motif3': [lambda G,timer: motif(G, level=3,timer=timer), lambda G: compare_motifs(G, level=3)]
    # 'Motif4': [lambda G,timer: motif(G, level=4,timer=timer), lambda G: compare_motifs(G, level=4)]
    #	'Motif3_GPU': {'cpp':lambda G,timer: motif(G, level=3,timer=timer),'gpu':lambda G,timer: motif(G, level=3,timer=timer,gpu=True)},
    #	'Motif4_GPU': {'cpp':lambda G,timer: motif(G, level=4,timer=timer),'gpu':lambda G,timer: motif(G, level=4,timer=timer,gpu=True)},
    'attraction_basin':[attraction_basin,lambda G: AttractorBasinCalculator(G).build()],
    'flow':[flow,lambda G: FlowCalculator(G).build()]
    
}

SEED = 123456

def benchmark_gpu_feature_regular_graph(feature_name, type='cpp', d=20):
    assert type in ['cpp','python','gpu']
    nodes = [50, 100, 500, 1000, 2000, 5000, 10000,100000]
    if type is 'python':
        titles = ['Feature calculation time']
    else:
        titles = ['Conversion Time', 'Feature calculation time']
	
    feature_type = type
    timer = FeatureTimer(feature_name + '_{}_benchmark'.format(feature_type),
                         titles=titles)

    for n in nodes:
        G = nx.random_regular_graph(d, n,seed=SEED,directed=True)
        run_id = '{}_nodes_and_{}_edges'.format(n, G.size())
        print(run_id)
        timer.start(run_id)
        if type is 'python':
            matching_python_functions[feature_name][type](G)
        else:
            matching_python_functions[feature_name][type](G, timer=timer)

        if type is 'python':
            timer.stop()



def benchmark_feature_regular_graph(feature_name, python=False, d=20):
    nodes = [50, 100, 500, 1000, 2000, 5000, 10000,100000]
    if python:
        feature_type = 'python'
        titles = ['Feature calculation time']
    else:
        feature_type = 'cpp'
        titles = ['Conversion Time', 'Feature calculation time']

    timer = FeatureTimer(feature_name + '_{}_benchmark'.format(feature_type),
                         titles=titles)

    for n in nodes:
        G = nx.random_regular_graph(d, n,seed=SEED)
        run_id = '{}_nodes_and_{}_edges'.format(n, G.size())
        print(run_id)
        timer.start(run_id)
        if python:
            matching_python_functions[feature_name][1](G)
        else:
            matching_python_functions[feature_name][0](G, timer=timer)

        if python:
            timer.stop()


def benchmark_feature_erdos_renyi(feature_name, python=False, p=0.8):
    nodes = [50, 100, 300, 500, 1000, 2000, 3000, 4000, 5000]

    if python:
        feature_type = 'python'
        titles = ['Feature calculation time']
    else:
        feature_type = 'cpp'
        titles = ['Conversion Time', 'Feature calculation time']

    timer = FeatureTimer(feature_name + '_{}_benchmark'.format(feature_type),
                         titles=titles)

    for n in nodes:
        G = nx.erdos_renyi_graph(n, p, seed=SEED)
        run_id = '{}_nodes_and_{}_edges'.format(n, G.size())
        print(run_id)
        timer.start(run_id)
        if python:
            matching_python_functions[feature_name][1](G)
        else:
            matching_python_functions[feature_name][0](G, timer=timer)

        if python:
            timer.stop()

def run_all_feature_with_gpu_tests_regular_graphs():
	for feature_name in matching_python_functions.keys():
		print(feature_name)
		benchmark_gpu_feature_regular_graph(feature_name,type='cpp')
		benchmark_gpu_feature_regular_graph(feature_name,type='gpu')

			
def run_all_feature_tests_regular_graphs():
    for feature_name in matching_python_functions.keys():
        print(feature_name)
        benchmark_feature_regular_graph(feature_name, python=False)
        benchmark_feature_regular_graph(feature_name, python=True)


def run_all_feature_tests_erdos_renyi():
    for feature_name in matching_python_functions.keys():
        print(feature_name)
        benchmark_feature_erdos_renyi(feature_name, python=False)
        benchmark_feature_erdos_renyi(feature_name, python=True)


if __name__ == '__main__':
    # run_all_feature_with_gpu_tests_regular_graphs()
    run_all_feature_tests_regular_graphs()
