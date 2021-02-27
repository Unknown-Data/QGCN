/*
 * WrapperIncludes.cpp
 *
 *  Created on: Nov 20, 2018
 *      Author: ori
 */
#include "WrapperIncludes.h"



py::list convertVectorOfVectorsTo2DList(vector<vector<unsigned int>*>* vec) {
	py::list mainList;
	for (auto l : *vec) {
		mainList.append(vectorToPythonList<unsigned int>(*l));
	}
	return mainList;

}

