/*
 * NodePageRankFeatureCalculator.h
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#ifndef FEATURES_NODEPAGERANKFEATURECALCULATOR_H_
#define FEATURES_NODEPAGERANKFEATURECALCULATOR_H_

#include "FeatureCalculator.h"
#include <vector>

class NodePageRankFeatureCalculator: public FeatureCalculator<std::vector<float>> {
public:
	NodePageRankFeatureCalculator(float dumping,unsigned int numOfIterations);
	virtual std::vector<float> Calculate();
	virtual ~NodePageRankFeatureCalculator();

private:
	float dumping;
	unsigned int numOfIterations;
};

#endif /* FEATURES_NODEPAGERANKFEATURECALCULATOR_H_ */
