/*
 * AttractionBasinCalculator.h
 *
 *  Created on: Jan 14, 2019
 *      Author: ori
 */

#ifndef FEATURES_ATTRACTIONBASINCALCULATOR_H_
#define FEATURES_ATTRACTIONBASINCALCULATOR_H_

#include "FeatureCalculator.h"
#include "DistanceUtils.h"
#include <vector>
#include <map>
#include <math.h>

class AttractionBasinCalculator: public FeatureCalculator<std::vector<double>*> {
public:
	AttractionBasinCalculator(int alpha);
	AttractionBasinCalculator();

	virtual std::vector<double>* Calculate();
	virtual ~AttractionBasinCalculator();

private:
	void calc_attraction_basin_dists();
	void calc_average_per_dist();
	void printVars();

	// The exponential decent coefficient
	int alpha;

	// For each node, we count the number of occurrences of each dist.
	// Hence, ab_*_dist[n][d] will give the number of nodes that are of a distance d from n.
	std::vector<std::map<unsigned int, unsigned int>*>* ab_in_dist;
	std::vector<std::map<unsigned int, unsigned int>*>* ab_out_dist;

	// For each distance, save the average number of nodes in that distance (over the entire graph).
	std::map<unsigned int,double>* average_out_per_dist;
	std::map<unsigned int, double>* average_in_per_dist;

	// The feature list to return
	std::vector<double>* features;
};

#endif /* FEATURES_ATTRACTIONBASINCALCULATOR_H_ */
