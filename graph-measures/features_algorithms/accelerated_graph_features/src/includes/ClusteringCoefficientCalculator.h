#pragma once
#include "stdafx.h"
#include "FeatureCalculator.h"

/*
	Calculate the average clustering coefficient of a network.
	We assume the network to be UNDIRECTED
*/
class ClusteringCoefficientCalculator :
	public FeatureCalculator<float>
{
public:
	ClusteringCoefficientCalculator();
	virtual float Calculate();
	
	virtual bool checkGPUEnabled(){
		return false;
	};
	virtual ~ClusteringCoefficientCalculator();
private:
	float LocalClusteringCoefficient(unsigned int i, unsigned int begin_offset, unsigned int end_offset, const unsigned int * neighborList);
};

