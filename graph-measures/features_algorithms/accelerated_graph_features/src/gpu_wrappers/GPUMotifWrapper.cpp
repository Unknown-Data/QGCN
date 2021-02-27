/*
 * ExampleWrapper.cpp
 *
 *  Created on: Nov 11, 2018
 *      Author: ori
 */

#include "GPUMotifWrapper.h"

void BoostDefGPUMotifCalculator() {
	def("motif_gpu", GPUMotifCalculatorWrapper);
}


py::list GPUMotifCalculatorWrapper(dict converted_dict,int level) {
	bool directed = extract<bool>(converted_dict["directed"]);
	//	std::cout << directed <<std::endl;
	ConvertedGNXReciever reciever(converted_dict);
	GPUMotifCalculator calc(level, directed);
	calc.setGraph(reciever.getCacheGraph());
	vector<vector<unsigned int>*>* res = calc.Calculate();
	py::list motif_counters = convertVectorOfVectorsTo2DList(res);
	for (auto p : *res) {
		delete p;
	}
	delete res;
	return motif_counters;

}
