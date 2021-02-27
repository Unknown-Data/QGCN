/*
 * KCoreFeatureCalculator.h
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#ifndef FEATURES_KCOREFEATURECALCULATOR_H_
#define FEATURES_KCOREFEATURECALCULATOR_H_

#include <vector>
#include "FeatureCalculator.h"

class KCoreFeatureCalculator: public FeatureCalculator<std::vector<unsigned short>> {
public:
	KCoreFeatureCalculator();
	virtual std::vector<unsigned short> Calculate();
	virtual ~KCoreFeatureCalculator();
};

#endif /* FEATURES_KCOREFEATURECALCULATOR_H_ */
