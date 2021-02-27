import sys
import os

# Leave the path changes here!!!
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import networkx as nx
import matplotlib.pyplot as plt
from src.accelerated_graph_features.test_python_converter import create_graph

N = 3

def plot_graph(i):
    G = create_graph(i)
    pos = nx.spring_layout(G) 
    nx.draw(G,pos)
    # labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    for i in range(1,N+1):
        plot_graph(i)
