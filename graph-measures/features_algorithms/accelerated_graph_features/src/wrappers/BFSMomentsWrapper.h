/*
 * BFSMomentsWrapper.h
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#ifndef WRAPPERS_BFSMOMENTSWRAPPER_H_
#define WRAPPERS_BFSMOMENTSWRAPPER_H_

#include "WrapperIncludes.h"
#include "../includes/BfsMomentsCalculator.h"
#include <tuple>

void BoostDefBFSMoments();

py::list BFSMomentWrapper(dict converted_dict);


#endif /* WRAPPERS_BFSMOMENTSWRAPPER_H_ */
