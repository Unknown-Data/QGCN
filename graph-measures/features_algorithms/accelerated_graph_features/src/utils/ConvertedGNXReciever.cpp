/*
 * ConvertedGNXReciever.cpp
 *
 *  Created on: Oct 28, 2018
 *      Author: ori
 */

#include "../includes/ConvertedGNXReciever.h"

ConvertedGNXReciever::ConvertedGNXReciever(dict converted_graph) {

	list offsetList = extract<list>(converted_graph["indices"]);
	list neighborList = extract<list>(converted_graph["neighbors"]);
	bool withWeights = extract<bool>(converted_graph["with_weights"]);
	bool directed = extract<bool>(converted_graph["directed"]);
	list weightsList;
	if(withWeights)
		weightsList = extract<list>(converted_graph["weights"]);


	this->offsets = new std::vector<int64>();
	this->offsets->reserve(len(offsetList));
	this->neighbors = new std::vector<unsigned int>();
	this->neighbors->reserve(len(neighborList));
	this->weights = new std::vector<double>();

//	std::cout << "Offset List:" << std::endl;
	for (int i = 0; i < len(offsetList); ++i) {
//		std::cout << extract<int>(offsetList[i]) << std::endl;
		int64 currentOffset;
		currentOffset =
				static_cast<int64>(extract<unsigned int>(offsetList[i]));

		this->offsets->push_back(currentOffset);
	}

//	std::cout << "Neighbor List:" << std::endl;
	for (int i = 0; i < len(neighborList); ++i) {
//		std::cout << extract<int>(neighborList[i]) << std::endl;
		unsigned int currentNeighbor = extract<unsigned int>(neighborList[i]);
		this->neighbors->push_back(currentNeighbor);
	}
	this->mGraph = new CacheGraph(directed);
	if(withWeights){
		for (int i = 0; i < len(weightsList); ++i) {
		//		std::cout << extract<int>(neighborList[i]) << std::endl;
				double currentNeighbor = extract<double>(weightsList[i]);
				this->weights->push_back(currentNeighbor);
			}
		mGraph->Assign(*offsets, *neighbors,*weights);

	}else{
		mGraph->Assign(*offsets, *neighbors);

	}



}

//ConvertedGNXReciever::ConvertedGNXReciever(const char* loadFileName) {
//
//	this->neighbors = NULL;
//	this->offsets = NULL;
//	this->mGraph = new CacheGraph();
//	mGraph->LoadFromFile(loadFileName);
//
//}
//
//ConvertedGNXReciever::ConvertedGNXReciever(dict converted_graph,
//		const char* saveFileName) {
//	this->neighbors = NULL;
//	this->offsets = NULL;
//	this->mGraph = NULL;
//	ConvertedGNXReciever(converted_graph);
//	this->mGraph->SaveToFile(saveFileName);
//
//}

ConvertedGNXReciever::~ConvertedGNXReciever() {

	delete offsets;
	delete neighbors;
	delete mGraph;
}

