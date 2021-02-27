/*
 * FlowCalculator.cpp
 *
 *  Created on: Jan 22, 2019
 *      Author: ori
 */

#include "../includes/FlowCalculator.h"

FlowCalculator::FlowCalculator(double threshold) :
		numOfNodes(0), threshold(threshold), features(NULL) {

}
void FlowCalculator::init() {
	numOfNodes = mGraph->GetNumberOfNodes();

	mGraph->InverseGraph(inverse);
	mGraph->CureateUndirectedGraph(inverse, undirectedGraph);

	directed_dists.reserve(numOfNodes);

	undirected_dists.reserve(numOfNodes);

	b_u.reserve(numOfNodes);
	features = new std::vector<double>();
	features->reserve(numOfNodes);
}

void FlowCalculator::CalcDists() {
	for (unsigned int node = 0; node < numOfNodes; node++) {
		auto directedBfsDist = DistanceUtils::BfsSingleSourceShortestPath(
				mGraph, node);
		directed_dists.push_back(
				new std::vector<unsigned int>(directedBfsDist.begin(),
						directedBfsDist.end()));

		auto undirectedBfsDist = DistanceUtils::BfsSingleSourceShortestPath(
				&undirectedGraph, node);
		undirected_dists.push_back(
				new std::vector<unsigned int>(undirectedBfsDist.begin(),
						undirectedBfsDist.end()));

	}
}

void FlowCalculator::CountReachables() {

	for (unsigned int src = 0; src < numOfNodes; src++) {
		b_u[src] = 0;
		for (unsigned int dest = 0; dest < numOfNodes; dest++) {
			if (directed_dists[src]->at(dest) > 0)
				b_u[src]++;
			else if (directed_dists[dest]->at(src) > 0) {
				b_u[src]++;
			}
		}
	}

}

void FlowCalculator::printVars() {
	std::cout << "directed dists" << std::endl;
	for (auto node_vec : directed_dists) {
		for (auto d : *node_vec)
			std::cout << d << "\t";
		std::cout << std::endl;
	}

	std::cout << "undirected dists" << std::endl;
	for (auto node_vec : undirected_dists) {
		for (auto d : *node_vec)
			std::cout << d << "\t";
		std::cout << std::endl;
	}

	std::cout << "b_u" << std::endl;
	for (int i = 0; i < numOfNodes; i++)
		std::cout << i << " : " << b_u[i] << std::endl;
}

std::vector<double>* FlowCalculator::Calculate() {
	CalcDists();
	CountReachables();

//	printVars();

	double max_b_u = (double) (*std::max_element(b_u.begin(), b_u.end()));

	for (unsigned int node = 0; node < numOfNodes; node++) {
		features->push_back(0);
		// Check threshold
		if ((b_u[node] / max_b_u) <= threshold) {
			continue;
		}
		auto udists = undirected_dists[node];
		auto dists = directed_dists[node];

		double sum = 0;

		for (unsigned int n = 0; n < numOfNodes; n++) {
			if (dists->at(n) == 0) {
				continue;
			}

			sum += (double) udists->at(n) / dists->at(n);
			(*features)[node] = sum / (double) b_u[node];
		}

	}
	return features;

}

FlowCalculator::~FlowCalculator() {
	for (auto p : directed_dists)
		delete p;

	for (auto p : undirected_dists)
		delete p;
}

