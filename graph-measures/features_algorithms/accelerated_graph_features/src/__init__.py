import sys
import os
# Leave the path changes here!!!
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# print('Changing to ' + os.path.dirname(__file__))
# os.chdir(os.path.dirname(__file__))
# print(os.getcwd())
from features_algorithms.accelerated_graph_features.src.accelerated_graph_features.feature_wrappers import *
