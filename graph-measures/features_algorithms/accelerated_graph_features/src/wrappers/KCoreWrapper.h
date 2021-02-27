/*
 * KCoreWrapper.h
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#ifndef WRAPPERS_KCOREWRAPPER_H_
#define WRAPPERS_KCOREWRAPPER_H_

#include "WrapperIncludes.h"
#include "../includes/KCoreFeatureCalculator.h"


void BoostDefKCore();
boost::python::list KCoreCalculatorWrapper(dict converted_graph);



#endif /* WRAPPERS_KCOREWRAPPER_H_ */
