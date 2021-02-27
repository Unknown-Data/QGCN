/*
 * ExampleFeatureCalculator.h
 *
 *  Created on: Oct 29, 2018
 *      Author: ori
 */

#ifndef EXAMPLEFEATURECALCULATOR_H_
#define EXAMPLEFEATURECALCULATOR_H_

#include "FeatureCalculator.h"

// Notice that we're returning a float from this feature calculation,
// it could just as easily have been a vector<float>

class ExampleFeatureCalculator: public FeatureCalculator<float> {

public:
	ExampleFeatureCalculator();
	void printOffsets();
	void printNodeRanks();
	virtual void setGraph(const CacheGraph* g) {
		this->mGraph = g;
	}
	// This is the only function you actually need to implement!
	virtual float Calculate();
	virtual ~ExampleFeatureCalculator();
private:
	// You can add whatever private variables\methods you want!

};

#endif /* EXAMPLEFEATURECALCULATOR_H_ */
