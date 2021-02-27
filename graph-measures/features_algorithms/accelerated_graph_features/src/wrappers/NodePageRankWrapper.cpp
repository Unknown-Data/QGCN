/*
 * NodePageRankWrapper.cpp
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#include "NodePageRankWrapper.h"

void BoostDefNodePageRank() {
	def("node_page_rank", NodePageRankWrapper);
}

py::list NodePageRankWrapper(dict converted_graph, float dumping, unsigned int numOfIterations) {
	ConvertedGNXReciever reciever(converted_graph);
	NodePageRankFeatureCalculator calc(dumping,numOfIterations);
	calc.setGraph(reciever.getCacheGraph());
	std::vector<float> res = calc.Calculate();
	return vectorToPythonList<float>(res);
}
