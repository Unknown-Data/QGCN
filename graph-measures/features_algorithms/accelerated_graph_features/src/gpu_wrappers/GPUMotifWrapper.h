/*
 * ExampleWrapper.h
 *
 *  Created on: Nov 11, 2018
 *      Author: ori
 */

#ifndef WRAPPERS_GPU_MOTIF_WRAPPER_H_
#define WRAPPERS_GPU_MOTIF_WRAPPER_H_

#include "../wrappers/WrapperIncludes.h"
#include "../includes/GPUMotifCalculator.h"

void BoostDefGPUMotifCalculator();
py::list GPUMotifCalculatorWrapper(dict converted_graph,int level);


#endif /* WRAPPERS_EXAMPLEWRAPPER_H_ */
