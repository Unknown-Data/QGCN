import os
import sys

import pickle
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt

# LEAVE THE PATH CHANGES HERE!!!
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(os.path.dirname(CUR_PATH))
VERBOSE = True


class MotifDrawer:
    def __init__(self, level: int = 3, directed: bool = True):
        assert level in [3, 4], "Unsupported motif level {}".format(level)
        self._level = level
        self._directed = directed
        self._motif_variations = self._load_motif_variations()

    def _load_motif_variations(self):
        fname = "%d_%sdirected.pkl" % (self._level, "" if self._directed else "un")
        fpath = os.path.join(BASE_PATH, 'features_algorithms', "motif_variations", fname)
        return pickle.load(open(fpath, "rb"))

    def draw_motif(self, index):
        if VERBOSE:
            print(index)
        if hasattr(index, '__iter__'):
            for idx in index:
                self.draw_motif(idx)
            return

        motif_number = max([x for x in self._motif_variations.keys() if self._motif_variations[x] == index])
        if VERBOSE:
            print(motif_number)

        bitmask = bin(motif_number)[2:][::-1]
        connection_list = [int(b) for b in bitmask] + [0] * ((6 if self._level == 3 else 12) - len(bitmask))

        motif_adg_matrix = []
        for _ in range(self._level):
            motif_adg_matrix.append([0] * self._level)

        pos = 0  # Current position in the connection_list

        if self._directed:
            for i, j in product(range(self._level), repeat=2):
                if VERBOSE:
                    print(pos, ' : ', i, j)
                if i == j:
                    continue
                motif_adg_matrix[i][j] = connection_list[pos]
                pos += 1
        else:
            for i in range(self._level):
                for j in range(i + 1, self._level):
                    if VERBOSE:
                        print(pos, ' : ', i, j)
                    motif_adg_matrix[i][j] = motif_adg_matrix[j][i] = connection_list[pos]
                    pos += 1

        # Build the graph
        if self._directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_edges_from([(i, j) for i, j in product(range(self._level), repeat=2) if motif_adg_matrix[i][j] != 0])

        # Draw it
        pos = nx.layout.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        if self._directed:
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=30)
        else:
            nx.draw_networkx_edges(G, pos)

        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

        plt.title('Motif with index {}'.format(index))
        plt.axis('off')
        plt.show()

        pass


if __name__ == '__main__':
    md = MotifDrawer(level=3, directed=True)
    md.draw_motif([10])
    # md.draw_motif([3, 4, 15,17, 23, 24, 80])
