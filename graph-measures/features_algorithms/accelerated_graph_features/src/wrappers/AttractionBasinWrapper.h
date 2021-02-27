/*
 * AttractionBasinWrapper.h
 *
 *  Created on: Jan 15, 2019
 *      Author: cohen
 */

#ifndef SRC_WRAPPERS_ATTRACTIONBASINWRAPPER_H_
#define SRC_WRAPPERS_ATTRACTIONBASINWRAPPER_H_


#include "WrapperIncludes.h"
#include "../includes/AttractionBasinCalculator.h"
void BoostDefAttractionBasinCalculator();
py::list AttractionBasinCalculatorWrapper(dict converted_graph,int alpha=2);


#endif /* SRC_WRAPPERS_ATTRACTIONBASINWRAPPER_H_ */
