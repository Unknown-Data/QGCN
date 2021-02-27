#include "../includes/MathUtils.h"

float MathUtils::calculateStd(const std::vector<float>& data) {
	float standartDeviation = 0.0f;
	int len = data.size();
	int nonZero = 0;
	float mean = calculateMeanWithoutZeroes(data);
	for (int i = 0; i < len; i++)
		if (data[i] != 0) {
			nonZero++;
			standartDeviation += (data[i] - mean) * (data[i] - mean);
		}

	standartDeviation = sqrt(standartDeviation / ((float) nonZero));

	return standartDeviation;
}

float MathUtils::calculateMean(const std::vector<float>& data) {
	int len = data.size();
	float sum = 0.0f;
	for (int i = 0; i < len; i++) {
		sum += data[i];
	}
	return sum / ((float) len);
}

float MathUtils::calculateMeanWithoutZeroes(const std::vector<float>& data) {

	int len = data.size();
	int nonZero = 0;
	float sum = 0.0f;
	for (int i = 0; i < len; i++) {
		if (data[i] != 0) {
			sum += data[i];
			nonZero++;
		}
	}
	return sum / ((float) nonZero);

}

float MathUtils::calculateWeightedAverage(const std::vector<float>& data,
		const std::vector<float>& weights) {

	int lenData = data.size();
	int lenWeights = weights.size();

	if (lenData != lenWeights)
		throw std::length_error("Data and weights must have the same size");

	float sum = 0.0f;
	for (int i = 0; i < lenData; i++)
		sum += data[i] * weights[i];

	float weightSum = 0;
	for (int i = 0; i < lenWeights; i++)
		weightSum += weights[i];

	sum = sum / weightSum;
	return sum;
}

float MathUtils::calculateWeightedStd(const std::vector<float>& data,
		const std::vector<float>& weights) {

	int lenData = data.size();
	int lenWeights = weights.size();

	if (lenData != lenWeights)
		throw std::length_error("Data and weights must have the same size");

	float avg = calculateWeightedAverage(data,weights);
	std::vector<float> modified_data;
	modified_data.reserve(lenData);
	for(auto& p:data)
		modified_data.push_back((p-avg)*(p-avg));
	float variance = calculateWeightedAverage(modified_data,weights);
	return sqrt(variance);


}
