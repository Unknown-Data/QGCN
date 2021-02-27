import os
import sys

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('../../../..'))
sys.path.append(os.path.abspath('../../../../..'))
import matplotlib.pyplot as plt
import numpy as np
from src.accelerated_graph_features.utils.data_reader import get_number_data


def plot_line_log_scale(feature_name, python_file, cpp_file, gpu_file):
    """
    Plot a line of the performance results in a log/log scale with a line (not bars)
    :param python_file: the file with the python benchmarks
    :param cpp_file: the file with the cpp benchmarks
    :param gpu_file:the file with the gpu benchmarks
    :return:
    """

    python_results = get_number_data(python_file)
    cpp_results = get_number_data(cpp_file)
    gpu_results = get_number_data(gpu_file)

    python_feature_time = np.asarray([d['Feature calculation time'] / 10 ** 6 for d in python_results])
    pf = np.log2(python_feature_time)
    cpp_feature_time = np.asarray([d['Feature calculation time'] / 10 ** 6 for d in cpp_results])
    cf = np.log2(cpp_feature_time)
    gpu_feature_time = np.asarray([d['Feature calculation time'] / 10 ** 6 for d in gpu_results])
    gf = np.log2(gpu_feature_time)

    X = np.asarray([float(d['run id'].split('_')[0]) for d in python_results])
    X = np.log2(X)
    N = len(cpp_results)

    py_plot = plt.plot(X[:len(python_results)], pf, color='green')
    cpp_plot = plt.plot(X[:len(cpp_results)], cf, color='orange')
    gpu_plot = plt.plot(X[:len(gpu_results)], gf, color='red')
    plt.ylabel(' log(Time[s]) ')
    plt.xlabel(' log(Nodes) ')
    plt.title('Feature Time Comparison for ' + feature_name.capitalize())

    plt.legend((py_plot[0], cpp_plot[0], gpu_plot[0]),
               ('Python', 'C++', 'GPU'))

    plt.show()


def plot_gpu_benchmark_comparison(feature_name):
    cpp_file = feature_name + '_GPU_cpp_benchmark.csv'
    gpu_file = feature_name + '_GPU_gpu_benchmark.csv'

    cpp_results = get_number_data(cpp_file)
    gpu_results = get_number_data(gpu_file)

    cpp_feature_time = [d['Feature calculation time'] / 10 ** 6 for d in cpp_results]
    cf = cpp_feature_time

    gpu_feature_time = [d['Feature calculation time'] / 10 ** 6 for d in gpu_results]
    gf = gpu_feature_time

    runs = [d['run id'] for d in gpu_results]

    N = len(cpp_results)

    X = np.arange(N)
    width = 0.2

    # Plot bar chart
    plt.figure(1)

    cpp_feature_bar = plt.bar(X + width, cf, width, color='orange')
    gpu_feature_bar = plt.bar(X, gf, width, color='red')

    plt.ylabel('Time')
    plt.title('Feature Time Comparison for ' + feature_name.capitalize())
    plt.xticks(X, runs, rotation=90)
    plt.legend((cpp_feature_bar[0], gpu_feature_bar[0]),
               ('C++ Feature', 'GPU Feature'))

    plt.show()


def plot_benchmark_comparison(feature_name):
    cpp_file = feature_name + '_cpp_benchmark.csv'
    python_file = feature_name + '_python_benchmark.csv'

    cpp_results = get_number_data(cpp_file)
    python_results = get_number_data(python_file)

    cpp_conversion_time = [d['Conversion Time'] / 10 ** 6 for d in cpp_results]
    cc = cpp_conversion_time
    cpp_feature_time = [d['Feature calculation time'] / 10 ** 6 for d in cpp_results]
    cf = cpp_feature_time

    python_feature_time = [d['Feature calculation time'] / 10 ** 6 for d in python_results]
    pf = python_feature_time

    runs = [d['run id'] for d in python_results]

    N = len(cpp_results)

    X = np.arange(N)
    width = 0.2

    # Plot bar chart
    plt.figure(1)

    cpp_conversion_bar = plt.bar(X, cc, width)
    cpp_feature_bar = plt.bar(X, cf, width, bottom=cc)
    python_feature_bar = plt.bar(X + width, pf, width)

    plt.ylabel('Time')
    plt.title('Feature Time Comparison for ' + feature_name.capitalize())
    plt.xticks(X, runs, rotation=90)
    plt.legend((cpp_conversion_bar[0], cpp_feature_bar[0], python_feature_bar[0]),
               ('C++ Conversion', 'C++ Feature', 'Python Feature'))

    # Plot difference line plot
    plt.figure(2)

    total_difference = [pf[i] - (cc[i] + cf[i]) for i in range(N)]
    feature_difference = [pf[i] - cf[i] for i in range(N)]

    plt.plot(total_difference, label='Total difference')
    plt.plot(feature_difference, label='Feature Difference')
    plt.ylabel('Time')
    plt.title('Feature Time Difference for ' + feature_name.capitalize())
    plt.legend()

    plt.show()


if __name__ == '__main__':

    features = ['Motif3', 'Motif4']
    # features = ['flow']
    # features = ['clustering', 'k_core', 'page_rank']

    for f in features:
        # plot_benchmark_comparison(f)
        # plot_gpu_benchmark_comparison(f)
        plot_line_log_scale(f, '{}_python_benchmark.csv'.format(f),
                            '{}_cpp_benchmark.csv'.format(f),
                            '{}_GPU_gpu_benchmark.csv'.format(f))
