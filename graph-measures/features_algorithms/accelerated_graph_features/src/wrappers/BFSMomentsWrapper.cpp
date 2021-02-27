/*
 * BFSMomentsWrapper.cpp
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#include "BFSMomentsWrapper.h"

void BoostDefBFSMoments() {
	def("bfs_moments",BFSMomentWrapper);
}

py::list tupleVectorToPythonList(const std::vector<floatTuple>& v){
	py::list l;
//	std::cout<<"After list create"<<std::endl;
	for(int i=0;i<v.size();i++){
//		std::cout<<"In loop iter "<<i<<std::endl;
		std::tuple<float,float> current = v[i];

		py::tuple py_tuple = py::make_tuple<float,float>(std::get<0>(current),std::get<1>(current));
		l.append<py::tuple>(py_tuple);
	}

	return l;
}


py::list BFSMomentWrapper(dict converted_dict) {

	ConvertedGNXReciever reciever(converted_dict);
	BfsMomentsCalculator calc;
	calc.setGraph(reciever.getCacheGraph());
//	std::cout<<"After conversion"<<std::endl;
	std::vector<std::tuple<float, float>> resVec = calc.Calculate();
	return tupleVectorToPythonList(resVec);

}
