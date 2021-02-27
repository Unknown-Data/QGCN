/*
 * ConvertedGNXReciever.h
 *
 *  Created on: Oct 28, 2018
 *      Author: ori
 */

#ifndef UTILS_CONVERTEDGNXRECIEVER_H_
#define UTILS_CONVERTEDGNXRECIEVER_H_

#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python.hpp>
#include <vector>

#include "stdafx.h"
#include "CacheGraph.h"

using namespace boost::python;


/**
  This class has a few responsibilities:
  	  a) Convert a dictionary of lists (that was created by the Python converter) into a CacheGraph
  	  b) Memory management: this object also holds a pointer to the CacheGraph, and the graph will be
  	  	  	  	  deleted alongside to converter at the end of the converter's scope.
  	  	  	  	  This will usually occur at the end of the wrapper function (that is exposed to Python)
  	  	  	  	  and so the graph will be cleaned automatically at the end of the calculation.
 */
class ConvertedGNXReciever {
public:
	ConvertedGNXReciever(dict converted_graph);
//	ConvertedGNXReciever(dict converted_graph,const char* saveGraph);
//	ConvertedGNXReciever(const char* loadFileName);
	const CacheGraph* getCacheGraph(){return mGraph;};
	virtual ~ConvertedGNXReciever();

private:
	std::vector<int64>* offsets;
	std::vector<unsigned int>* neighbors;
	std::vector<double>* weights;
	CacheGraph* mGraph;


};

#endif /* UTILS_CONVERTEDGNXRECIEVER_H_ */
