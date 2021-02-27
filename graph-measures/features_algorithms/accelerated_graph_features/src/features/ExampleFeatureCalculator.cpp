/*
 * ExampleFeatureCalculator.cpp
 *
 *  Created on: Oct 29, 2018
 *      Author: ori
 */

#include "../includes/ExampleFeatureCalculator.h"

ExampleFeatureCalculator::ExampleFeatureCalculator() {
	// If there is any additional preprocessing you would like to perform,
	// this is the place for it

}

/**
 * The actual calculation goes here
 * For the example feature, we would just like to know that the graph has been passed down correctly into a CacheGraph instance,
 * and so we print out the nodes and neighbor lists as a validation.
 * This also has the added benefit of making sure that our standard code templates can run without errors.
 */
void ExampleFeatureCalculator::printOffsets() {
	const unsigned int* neighborList = mGraph->GetNeighborList();
	const int64* offsetList = mGraph->GetOffsetList();
	unsigned int numOfNodes = mGraph->GetNumberOfNodes();
	int64 begin_offset, end_offset;
	for (unsigned int i = 0; i < numOfNodes; i++) {
		begin_offset = offsetList[i];
		end_offset = offsetList[i + 1];
		for (auto p = neighborList + begin_offset;
				p < neighborList + end_offset; ++p) {
			//p is a pointer to the nodes in the adjacency list.
			//Example usage: the offset of the current node is offsetList[*p]
//			std::cout << offsetList[*p] << std::endl;
		}
	}
}
void ExampleFeatureCalculator::printNodeRanks() {
	std::vector<unsigned int> Degrees = mGraph->ComputeNodeDegrees();
	for (int i = 0; i < Degrees.size(); i++)
		std::cout<< i << ":" << Degrees[i] << std::endl;
}

float ExampleFeatureCalculator::Calculate() {
	printOffsets();
	printNodeRanks();
	return (float) mGraph->GetNumberOfNodes();
}

ExampleFeatureCalculator::~ExampleFeatureCalculator() {
// Don't forget to release any memory you may have allocated!
}

