import numpy as np
import networkx as nx
from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class HierarchyEnergyCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        # TODO: finish this calculator
        return False

    def _calculate(self, include: set):
        self._nodes_order = sorted(self._gnx)
        hierarchy_energy_list, vet_index = self._calculate_hierarchy_energy_index()
        self._features = dict(zip(vet_index, hierarchy_energy_list))

    def _calculate_hierarchy_energy_index(self):
        # vet_index, g = self._build_graph_matrix()
        adj = nx.adjacency_matrix(self._gnx, nodelist=self._nodes_order)
        l, y, tol, r, d = self._initialize_vars_from_laplacian_matrix(adj)
        # calculation of hierarchy Energy
        while np.linalg.norm(r) > tol:
            gamma = np.dot(r.T, r)
            alpha = np.divide(gamma, np.dot(d.T, np.dot(l, d)))
            y = np.add(y, alpha * d)
            r = np.subtract(r, alpha * np.dot(l, d))
            beta = np.divide((np.dot(r.T, r)), gamma)
            d = np.add(r, beta * d)
        return y, self._gnx.nodes()

    @staticmethod
    def _initialize_vars_from_laplacian_matrix(adj):
        from scipy import sparse
        # creating laplacian matrix
        # w = g + g.T
        # build symmetric adjacency matrix
        w = adj + (adj.T - adj).multiply(adj.T > adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d = sparse.diags(rowsum)
        # d = np.diag(np.sum(adj))
        laplacian = d - w

        i_d = np.sum(adj, 0)
        o_d = np.sum(adj, 1)
        # initialize_vars
        b = np.subtract((np.array([o_d])).T, (np.array([i_d])).T)
        tol = 0.001
        n = adj.shape[0]
        y = np.random.rand(n, 1)
        y = np.subtract(y, (1. / n) * sum(y))
        k = laplacian.dot(y)
        r = np.subtract(b, k)
        d = r
        return laplacian, y, tol, r, d


    @staticmethod
    def _initialize_vars_from_laplacian_matrix1(g):
        # creating laplacian matrix
        w = g + g.T
        d = np.diag(sum(w))
        l = d - w
        _id = np.sum(g, 0)
        od = np.sum(g, 1)
        # initialize_vars
        b = np.subtract((np.array([od])).T, (np.array([_id])).T)
        tol = 0.001
        n = np.size(g, 1)
        y = np.random.rand(n, 1)
        y = np.subtract(y, (1 / n) * sum(y))
        k = np.dot(l, y)
        r = np.subtract(b, k)
        d = r
        return l, y, tol, r, d


feature_entry = {
    "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),
}

if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(HierarchyEnergyCalculator, is_max_connected=True)
