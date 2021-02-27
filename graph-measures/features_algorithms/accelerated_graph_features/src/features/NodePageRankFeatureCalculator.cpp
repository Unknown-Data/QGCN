/*
 * NodePageRankFeatureCalculator.cpp
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#include "../includes/NodePageRankFeatureCalculator.h"

NodePageRankFeatureCalculator::NodePageRankFeatureCalculator(float dumping,
		unsigned int numOfIterations) :
		dumping(dumping), numOfIterations(numOfIterations) {

}

std::vector<float> NodePageRankFeatureCalculator::Calculate() {
	return mGraph->ComputeNodePageRank(this->dumping, this->numOfIterations);

}

NodePageRankFeatureCalculator::~NodePageRankFeatureCalculator() {

}

