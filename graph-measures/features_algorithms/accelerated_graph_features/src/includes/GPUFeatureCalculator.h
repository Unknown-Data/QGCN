#pragma once
#include "FeatureCalculator.h"
template<class T>
class GPUFeatureCalculator :
	public FeatureCalculator<T>
{
public:
	GPUFeatureCalculator();
	virtual ~GPUFeatureCalculator();
	virtual bool checkGPUEnabled();
	bool checkGPUAvailable();
};

