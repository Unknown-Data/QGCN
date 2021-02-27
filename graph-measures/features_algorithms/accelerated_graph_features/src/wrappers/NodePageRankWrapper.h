/*
 * NodePageRankWrapper.h
 *
 *  Created on: Nov 12, 2018
 *      Author: ori
 */

#ifndef WRAPPERS_NODEPAGERANKWRAPPER_H_
#define WRAPPERS_NODEPAGERANKWRAPPER_H_

#include "WrapperIncludes.h"
#include "../includes/NodePageRankFeatureCalculator.h"

void BoostDefNodePageRank();
py::list NodePageRankWrapper(dict converted_graph,float dumping, unsigned int numOfIterations);



#endif /* WRAPPERS_NODEPAGERANKWRAPPER_H_ */
