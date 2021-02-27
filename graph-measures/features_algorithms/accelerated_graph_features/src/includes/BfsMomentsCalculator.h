#ifndef BFS_MOMENTS_CALCULATOR_H_
#define BFS_MOMENTS_CALCULATOR_H_

#include "FeatureCalculator.h"
#include "CacheGraph.h"
#include "DistanceUtils.h"
#include "MathUtils.h"
#include <unordered_map>
#include <vector>
#include <tuple>


typedef std::tuple<float,float> floatTuple;

using namespace std;
class BfsMomentsCalculator :
	public FeatureCalculator<vector<floatTuple>>
{
public:
	BfsMomentsCalculator();
	virtual vector<floatTuple> Calculate();

	virtual ~BfsMomentsCalculator();

protected:
	virtual inline bool checkGPUEnabled() {
		return false;
	}
};

#endif
