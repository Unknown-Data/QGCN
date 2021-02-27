/*
 * FlowCalculator.h
 *
 *  Created on: Jan 22, 2019
 *      Author: ori
 */

#ifndef FEATURES_FLOWCALCULATOR_H_
#define FEATURES_FLOWCALCULATOR_H_

#include "FeatureCalculator.h"
#include "DistanceUtils.h"
#include <vector>
#include <map>
#include <set>
#include <algorithm>


class FlowCalculator: public FeatureCalculator<std::vector<double>*>{
public:
	FlowCalculator(double threshold=0);
	virtual std::vector<double>* Calculate();

	virtual ~FlowCalculator();

private:
	virtual void init();
	void CalcDists();
	void CountReachables();
	void printVars();

	double threshold;
	unsigned int numOfNodes;

	std::vector<double>* features;

	CacheGraph inverse,undirectedGraph;
	std::vector<std::vector<unsigned int>*> directed_dists;
	std::vector<std::vector<unsigned int>*> undirected_dists;
	std::vector<unsigned int> b_u;

};

#endif /* FEATURES_FLOWCALCULATOR_H_ */
