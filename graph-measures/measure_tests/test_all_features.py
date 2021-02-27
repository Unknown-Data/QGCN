from features_algorithms.vertices.attractor_basin import AttractorBasinCalculator
from features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
from features_algorithms.vertices.bfs_moments import BfsMomentsCalculator
from features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from features_algorithms.vertices.communicability_betweenness_centrality import \
    CommunicabilityBetweennessCentralityCalculator
from features_algorithms.vertices.eccentricity import EccentricityCalculator
from features_algorithms.vertices.fiedler_vector import FiedlerVectorCalculator
from features_algorithms.vertices.flow import FlowCalculator
from features_algorithms.vertices.general import GeneralCalculator
from features_algorithms.vertices.hierarchy_energy import HierarchyEnergyCalculator
from features_algorithms.vertices.k_core import KCoreCalculator
from features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from features_algorithms.vertices.louvain import LouvainCalculator
from features_algorithms.vertices.motifs import nth_edges_motif
from features_algorithms.vertices.page_rank import PageRankCalculator
from measure_tests.specific_feature_test import test_specific_feature


FEATURE_CLASSES = [
    AttractorBasinCalculator,
    AverageNeighborDegreeCalculator,
    BetweennessCentralityCalculator,
    BfsMomentsCalculator,
    ClosenessCentralityCalculator,
    CommunicabilityBetweennessCentralityCalculator,
    EccentricityCalculator,
    FiedlerVectorCalculator,
    FlowCalculator,
    GeneralCalculator,
    HierarchyEnergyCalculator,
    KCoreCalculator,
    LoadCentralityCalculator,
    LouvainCalculator,
    nth_edges_motif(3),
    PageRankCalculator,
]


def test_all():
    for cls in FEATURE_CLASSES:
        test_specific_feature(cls, is_max_connected=True)


if __name__ == "__main__":
    test_all()
