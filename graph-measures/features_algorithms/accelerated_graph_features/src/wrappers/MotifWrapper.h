/*
 * MotifWrapper.h
 *
 *  Created on: Dec 2, 2018
 *      Author: ori
 */

#ifndef WRAPPERS_MOTIFWRAPPER_H_
#define WRAPPERS_MOTIFWRAPPER_H_

#include "WrapperIncludes.h"
#include "../includes/MotifCalculator.h"

void BoostDefMotif();

py::list MotifCalculatorWrapper(dict converted_dict,int level);





#endif /* WRAPPERS_MOTIFWRAPPER_H_ */
