/*
 * ExampleWrapper.cpp
 *
 *  Created on: Nov 11, 2018
 *      Author: ori
 */

#include "ExampleWrapper.h"

void BoostDefExampleCalculator() {
	def("example_feature",ExampleCalculatorWrapper);
}

float ExampleCalculatorWrapper(dict converted_graph) {
	ConvertedGNXReciever reciever(converted_graph);
	ExampleFeatureCalculator calc;
	calc.setGraph(reciever.getCacheGraph());
	return calc.Calculate();
// At this point in time the graph is also deleted
}
