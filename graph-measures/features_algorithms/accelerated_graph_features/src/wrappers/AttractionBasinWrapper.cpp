/*
 * AttractionBasinWrapper.cpp
 *
 *  Created on: Jan 15, 2019
 *      Author: cohen
 */

#include "AttractionBasinWrapper.h"


void BoostDefAttractionBasinCalculator() {
	def("attraction_basin",AttractionBasinCalculatorWrapper);
}
py::list AttractionBasinCalculatorWrapper(dict converted_graph,int alpha){
	ConvertedGNXReciever reciever(converted_graph);
	AttractionBasinCalculator calc(alpha);
	calc.setGraph(reciever.getCacheGraph());
	std::vector<double>* res = calc.Calculate();
	auto ret_list = vectorToPythonList<double>(*res);
	delete res;
	return ret_list;
}


