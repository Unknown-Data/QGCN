/*
 * KCoreFeatureCalculator.cpp
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#include "../includes/KCoreFeatureCalculator.h"

KCoreFeatureCalculator::KCoreFeatureCalculator() {


}

std::vector<unsigned short> KCoreFeatureCalculator::Calculate() {

	return mGraph->ComputeKCore();

}

KCoreFeatureCalculator::~KCoreFeatureCalculator() {
}

