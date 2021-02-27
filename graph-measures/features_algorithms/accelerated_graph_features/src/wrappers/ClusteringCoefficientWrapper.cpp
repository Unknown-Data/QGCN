/*
 * ClusteringCoefficientWrapper.cpp
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#include "ClusteringCoefficientWrapper.h"

void BoostDefClusteringCoefficient() {
	def("clustering_coefficient",ClusteringCoefWrapper);
}

float ClusteringCoefWrapper(dict converted_dict) {
	ConvertedGNXReciever reciever(converted_dict);
	ClusteringCoefficientCalculator calc;
	calc.setGraph(reciever.getCacheGraph());
	return calc.Calculate();
}

