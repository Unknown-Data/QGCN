#pragma once

#include <vector>
#include "stdafx.h"
#include "CacheGraph.h"

using namespace std;

template<typename T> class FeatureCalculator
{
public:
	FeatureCalculator();
	virtual T Calculate() = 0;
	virtual ~FeatureCalculator();
	virtual const CacheGraph* getGraph();
	virtual void setGraph(const CacheGraph* graph);


protected:
	virtual void init();
	bool checkGPUEnabled(){
		return false;
	};
	const CacheGraph* mGraph;

private:
	bool isGPUEnabled;
};

template<class T>
FeatureCalculator<T>::FeatureCalculator():
	mGraph(NULL),isGPUEnabled(false)
{

}

template<typename T>
void FeatureCalculator<T>::setGraph(const CacheGraph* graph)
{
	this->mGraph = graph;
	init();
}

template<typename T>
void FeatureCalculator<T>::init()
{
	isGPUEnabled = checkGPUEnabled();
}

template<class T>
FeatureCalculator<T>::~FeatureCalculator()
{
}

template<typename T>
const CacheGraph* FeatureCalculator<T>::getGraph()
{
	return this->mGraph;
}


