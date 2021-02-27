#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_
#include <vector>
#include <cmath>
#include <stdexcept>

class MathUtils
{
public:
	static float calculateStd(const std::vector<float>& data);
	static float calculateMean(const std::vector<float>& data);
	static float calculateMeanWithoutZeroes(const std::vector<float>& data);
	static float calculateWeightedAverage(const std::vector<float>& data, const std::vector<float>& weights);
	static float calculateWeightedStd(const std::vector<float>& data, const std::vector<float>& weights);

};

#endif
