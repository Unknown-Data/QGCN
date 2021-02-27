/*
 * MotifWrapper.cpp
 *
 *  Created on: Dec 2, 2018
 *      Author: ori
 */

#include "MotifWrapper.h"

void BoostDefMotif() {
	def("motif",MotifCalculatorWrapper);
}

py::list MotifCalculatorWrapper(dict converted_dict,int level) {
	bool directed = extract<bool>(converted_dict["directed"]);
//	std::cout << directed <<std::endl;
	ConvertedGNXReciever reciever(converted_dict);
	MotifCalculator calc(level,directed);
	calc.setGraph(reciever.getCacheGraph());
	vector<vector<unsigned int>*>* res = calc.Calculate();
	py::list motif_counters = convertVectorOfVectorsTo2DList(res);
	for(auto p:*res){
		delete p;
	}
	delete res;
	return motif_counters;

}
