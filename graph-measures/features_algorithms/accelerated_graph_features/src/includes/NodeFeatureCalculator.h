#pragma once
#include "stdafx.h"
#include "FeatureCalculator.h"

template<class T>
class NodeFeatureCalculator :
	public FeatureCalculator<T>
{
public:
	NodeFeatureCalculator();

	virtual ~NodeFeatureCalculator();
};

