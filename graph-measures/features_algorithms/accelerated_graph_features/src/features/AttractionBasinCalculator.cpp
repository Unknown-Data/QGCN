/*
 * AttractionBasinCalculator.cpp
 *
 *  Created on: Jan 14, 2019
 *      Author: ori
 */

#include "../includes/AttractionBasinCalculator.h"
#include <iostream>

AttractionBasinCalculator::AttractionBasinCalculator(int alpha) :
		alpha(alpha), ab_in_dist(NULL), ab_out_dist(NULL), average_out_per_dist(
		NULL), average_in_per_dist(NULL), features(NULL) {
}
AttractionBasinCalculator::AttractionBasinCalculator() :
		AttractionBasinCalculator(2) {
}

void AttractionBasinCalculator::printVars() {
	int i = 0;
	std::cout << "ab_out_dist" << std::endl;
	for (auto& m : *(this->ab_out_dist)) {
		std::cout << "Map at " << i++ << std::endl;
		for (auto& x : *m)
			std::cout << x.first << " : " << x.second << std::endl;

	}
	i = 0;
	std::cout << "ab_in_dist" << std::endl;
	for (auto& m : *(this->ab_in_dist)) {
		std::cout << "Map at " << i++ << std::endl;
		for (auto& x : *m)
			std::cout << x.first << " : " << x.second << std::endl;

	}

	std::cout << "avg_out" << std::endl;
	for (auto& x : *average_out_per_dist)
		std::cout << x.first << " : " << x.second << std::endl;

	std::cout << "avg_in" << std::endl;
	for (auto& x : *average_in_per_dist)
		std::cout << x.first << " : " << x.second << std::endl;
}

std::vector<double>* AttractionBasinCalculator::Calculate() {
	//std::cout << "Begin"<<std::endl;
	this->calc_attraction_basin_dists();
	//std::cout << "After dists"<<std::endl;
	this->calc_average_per_dist();
	//std::cout << "After avg"<<std::endl;
	unsigned int numOfNodes = mGraph->GetNumberOfNodes();
	//std::cout <<"alpha:"<<alpha<<std::endl;
	//printVars();

	features = new std::vector<double>();
	for (unsigned int node = 0; node < numOfNodes; node++) {
		auto out_dist = ab_out_dist->at(node);
		auto in_dist = ab_in_dist->at(node);

		features->push_back(-1);

		double numerator = 0, denominator = 0;
		for (auto& x : *out_dist) {
			auto dist = x.first;
			auto occurances = x.second;

			//	std::cout <<"\tdist: " <<dist<<std::endl;
			//	std::cout <<"\toccurances: " <<occurances<<std::endl;
			//	std::cout << "\tavg at dist: " <<average_out_per_dist->at(dist)<<std::endl;
			//	std::cout <<"\tpow: "<<1/(double) pow(alpha,dist)<<std::endl;
			denominator += (occurances / average_out_per_dist->at(dist))
					* (1 / (double) pow(alpha, dist));

		} //end summing loop

//	std::cout << "denom : "<<denominator<<std::endl;
		if (denominator != 0) {
			for (auto& x : *in_dist) {
				auto dist = x.first;
				auto occurances = x.second;

				numerator += (occurances / average_in_per_dist->at(dist))
						* (1 / (double) pow(alpha, dist));
			} //end summing loop
			(*features)[node] = numerator / denominator;
		} //end if
	}

	return this->features;
}

void AttractionBasinCalculator::calc_attraction_basin_dists() {

	unsigned int numOfNodes = mGraph->GetNumberOfNodes();
	this->ab_in_dist = new std::vector<std::map<unsigned int, unsigned int>*>();
	this->ab_out_dist =
			new std::vector<std::map<unsigned int, unsigned int>*>();

// Build a distance matrix
	std::vector<std::vector<unsigned int>*> dists;
	dists.reserve(numOfNodes);
	for (unsigned int node = 0; node < numOfNodes; node++) {
		auto bfsDist = DistanceUtils::BfsSingleSourceShortestPath(mGraph, node);
		dists.push_back(
				new std::vector<unsigned int>(bfsDist.begin(), bfsDist.end()));
		ab_out_dist->push_back(new std::map<unsigned int, unsigned int>());
		ab_in_dist->push_back(new std::map<unsigned int, unsigned int>());
	}

//std:cout << ab_out_dist->size()<<std::endl;

	for (unsigned int src = 0; src < numOfNodes; src++) {
		for (unsigned int dest = 0; dest < numOfNodes; dest++) {
			unsigned int d = dists.at(src)->at(dest);
			if (d > 0) {
				(*((*ab_out_dist)[src]))[d] =
						(ab_out_dist->at(src)->count(d) == 1) ?
								ab_out_dist->at(src)->at(d) + 1 : 1;

				(*(*ab_in_dist)[dest])[d] =
						(ab_in_dist->at(dest)->find(d)
								!= ab_in_dist->at(dest)->end()) ?
								ab_in_dist->at(dest)->at(d) + 1 : 1;

			} // end if
		}  // end dest loop
	} //end src loop

	for (auto& p : dists)
		delete p;
}

void AttractionBasinCalculator::calc_average_per_dist() {
	average_in_per_dist = new std::map<unsigned int, double>();
	average_out_per_dist = new std::map<unsigned int, double>();

	unsigned int numOfNodes = mGraph->GetNumberOfNodes();
	for (unsigned int src = 0; src < numOfNodes; src++) {

		// Unify the in distance counters
		auto counter = ab_in_dist->at(src);
		for (auto& x : *counter) {
			auto dist = x.first;
			auto occurances = x.second;
			(*average_in_per_dist)[dist] =
					(average_in_per_dist->find(dist)
							!= average_in_per_dist->end()) ?
							average_in_per_dist->at(dist) + occurances :
							occurances;
		}

		// Unify the out distances
		counter = ab_out_dist->at(src);
		for (auto& x : *counter) {
			auto dist = x.first;
			auto occurances = x.second;
			(*average_out_per_dist)[dist] =
					(average_out_per_dist->find(dist)
							!= average_out_per_dist->end()) ?
							average_out_per_dist->at(dist) + occurances :
							occurances;
		}

	} // End src loop

	for (auto& d : *average_out_per_dist) {
		(*average_out_per_dist)[d.first] = d.second / (double) numOfNodes;
	}
	for (auto& d : *average_in_per_dist) {
		(*average_in_per_dist)[d.first] = d.second / (double) numOfNodes;
	}

}

AttractionBasinCalculator::~AttractionBasinCalculator() {
	// TODO Auto-generated destructor stub

	for (int i = 0; i < this->ab_in_dist->size(); i++)
		delete ab_in_dist->at(i);

	delete ab_in_dist;

	for (int i = 0; i < this->ab_out_dist->size(); i++)
		delete ab_out_dist->at(i);
	delete ab_out_dist;

	delete average_out_per_dist;
	delete average_in_per_dist;

}
