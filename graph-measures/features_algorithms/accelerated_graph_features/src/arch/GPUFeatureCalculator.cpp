#include "stdafx.h"
#include "GPUFeatureCalculator.h"

template<class T>
GPUFeatureCalculator<T>::GPUFeatureCalculator()
{
}

template<class T>
GPUFeatureCalculator<T>::~GPUFeatureCalculator()
{
}

template<class T>
bool GPUFeatureCalculator<T>::checkGPUEnabled()
{
	if (/*check device has GPU (and CUDA?)*/ true)
		return true;
	return false;
}

template<typename T>
bool GPUFeatureCalculator<T>::checkGPUAvailable()
{
	//check that the device has a GPU
	return false;
}
