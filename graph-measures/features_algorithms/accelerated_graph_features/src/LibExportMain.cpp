/*
 * LibExportMain.cpp
 *
 *  Created on: Nov 11, 2018
 *      Author: ori
 */

#include <stdio.h>
#include <boost/python.hpp>


// Regular features
#include "wrappers/WrapperIncludes.h"
#include "wrappers/ExampleWrapper.h"
#include "wrappers/KCoreWrapper.h"
#include "wrappers/NodePageRankWrapper.h"
#include "wrappers/BFSMomentsWrapper.h"
#include "wrappers/ClusteringCoefficientWrapper.h"
#include "wrappers/MotifWrapper.h"
#include "wrappers/AttractionBasinWrapper.h"
#include "wrappers/FlowWrapper.h"

#ifdef __NVCC__
// GPU features
#include "gpu_wrappers/GPUMotifWrapper.h"
#endif
// ... other imports ...

/*
 * Check that exporting to Python works
 */
void test_export(){
	std::cout << "Hello Test!"<<std::endl;
}

BOOST_PYTHON_MODULE(_features)
{

	def("test",test_export);
// ... other boost def wrappers ...

	// Regular features
	BoostDefExampleCalculator();
	BoostDefKCore();
	BoostDefNodePageRank();
	BoostDefClusteringCoefficient();
	BoostDefBFSMoments();
	BoostDefMotif();
	BoostDefAttractionBasinCalculator();
	BoostDefFlowCalculator();

#ifdef __NVCC__	
	BoostDefGPUMotifCalculator();
#endif
}

