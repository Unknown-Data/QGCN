/*
 * AttractionBasinWrapper.cpp
 *
 *  Created on: Jan 15, 2019
 *      Author: cohen
 */

#include "FlowWrapper.h"


void BoostDefFlowCalculator() {
	def("flow",FlowCalculatorWrapper);
}

py::list FlowCalculatorWrapper(dict converted_graph, double threshold) {
	ConvertedGNXReciever reciever(converted_graph);
	FlowCalculator calc(threshold);
	calc.setGraph(reciever.getCacheGraph());
	std::vector<double>* res = calc.Calculate();
	auto ret_list = vectorToPythonList<double>(*res);
	delete res;
	return ret_list;
}
