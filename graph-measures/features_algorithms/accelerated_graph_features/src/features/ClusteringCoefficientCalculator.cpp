#include "../includes/ClusteringCoefficientCalculator.h"


ClusteringCoefficientCalculator::ClusteringCoefficientCalculator()
{
}

/*
	Calculate the average CC of the network
*/
float ClusteringCoefficientCalculator::Calculate()
{
	//The clustering average
	float clusteringCoef = 0;

	//Get the neighbors and offsets
	const unsigned int* neighborList = mGraph->GetNeighborList();
	const int64* offsetList = mGraph->GetOffsetList();
	
	//sum all of the local clustering coefficients
	unsigned int numOfNodes = mGraph->GetNumberOfNodes();
	int64 begin_offset, end_offset;
	for (unsigned int i = 0; i < numOfNodes; i++) {
		begin_offset = offsetList[i];
		end_offset = offsetList[i + 1];
		clusteringCoef += LocalClusteringCoefficient(i,begin_offset,end_offset,neighborList);
	}
	//return the average
	return clusteringCoef/numOfNodes;
}

/*
	A function to calculate the local CC of the node i.
	Input: the node index
	Output: the local CC of the node.
	Parameters:
		i - the index of the node for which we calculate the local CC
		begin_offset,end_offset - the beginning and ending indices of the section in the neighborhood list of i's neighbors
		neighborList - the list of neighbors for each node. Sorted by nodes and with the neighbors themselves sorted.

*/
float ClusteringCoefficientCalculator::LocalClusteringCoefficient(unsigned int i,unsigned int begin_offset, unsigned int end_offset, const unsigned int* neighborList) {

	//The Clustering Coefficient
	float cc = 0;

	//The nighborhood of i is now neighborList[begin:end]
	//Moreover, the size of that neighborhood is given by the following:
	unsigned int neighborhoodSize = static_cast<unsigned int>(end_offset - begin_offset);
	
	//This is the critical part of the formula. We need to count the number of neigbors that are also connected to each other.
	unsigned int connectedNeighbors = 0;

	for (auto p = neighborList + begin_offset; p < neighborList + end_offset; ++p)
	{
		//p is a pointer to the nodes in the adjacency list.
		//Example usage: the offset of the current node is offsetList[*p]
		for (auto q = neighborList + begin_offset; q < p; ++q) {
			//We need to go over all the pairs, but we don't want to count every edge twice, so we only iterate over q<p.
			if (mGraph->areNeighbors(*p, *q))
				connectedNeighbors++;
		}
	}

	//calculate the coefficient
	cc = (float) 2 * connectedNeighbors / (neighborhoodSize*(neighborhoodSize - 1));

	return cc;
}


ClusteringCoefficientCalculator::~ClusteringCoefficientCalculator()
{
}
