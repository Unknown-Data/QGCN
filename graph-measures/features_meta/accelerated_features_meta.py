from accelerated_graph_features.attractor_basin import AttractorBasinCalculator
from accelerated_graph_features.bfs_moments import BfsMomentsCalculator
from accelerated_graph_features.flow import FlowCalculator
from accelerated_graph_features.k_core import KCoreCalculator
from accelerated_graph_features.motifs import nth_nodes_motif
from accelerated_graph_features.page_rank import PageRankCalculator
from average_neighbor_degree import AverageNeighborDegreeCalculator
from betweenness_centrality import BetweennessCentralityCalculator
from closeness_centrality import ClosenessCentralityCalculator
from communicability_betweenness_centrality import CommunicabilityBetweennessCentralityCalculator
from eccentricity import EccentricityCalculator
from feature_calculators import FeatureMeta
from fiedler_vector import FiedlerVectorCalculator
from general import GeneralCalculator
from hierarchy_energy import HierarchyEnergyCalculator
from load_centrality import LoadCentralityCalculator
from louvain import LouvainCalculator


class FeaturesMeta:
    def __init__(self, gpu=False):
        self.NODE_LEVEL = {
            "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
            "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
            "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
            "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
            "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
            "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
                                                                  {"communicability"}),
            "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
            "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),
            "flow": FeatureMeta(FlowCalculator, {}),
            "general": FeatureMeta(GeneralCalculator, {"gen"}),
            "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),
            "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
            "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),
            "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
            "motif3": FeatureMeta(nth_nodes_motif(3, gpu), {"m3"}),
            "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
            "motif4": FeatureMeta(nth_nodes_motif(4, gpu), {"m4"}),
        }

        self.MOTIFS = {
            "motif3": FeatureMeta(nth_nodes_motif(3, gpu), {"m3"}),
            "motif4": FeatureMeta(nth_nodes_motif(4, gpu), {"m4"})
        }


