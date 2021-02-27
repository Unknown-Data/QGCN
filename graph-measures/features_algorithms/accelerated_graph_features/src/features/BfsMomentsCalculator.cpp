#include "../includes/BfsMomentsCalculator.h"

BfsMomentsCalculator::BfsMomentsCalculator() {
}

vector<floatTuple> BfsMomentsCalculator::Calculate() {

	const int numOfNodes = mGraph->GetNumberOfNodes();
	vector<floatTuple> features(numOfNodes);

	for (int i = 0; i < numOfNodes; i++) {
//		std::cout << "In main loop iter " << i << std::endl;

		//calculate BFS distances
		std::vector<unsigned int> distances = DistanceUtils::BfsSingleSourceShortestPath(
				mGraph, i);

//		std::cout<<"After DistanceUtils"<<std::endl;
//		std::cout<<"Distances: "<<std::endl;
//		for(int k=0;k<distances.size();k++)
//			std::cout<<k<<" "<<distances[k]<<std::endl;

//count the number of times each distance exists
		std::unordered_map<unsigned int, int> distCounter;

		for (int j = 0; j < distances.size(); j++) {
//			std::cout<<"In internal loop iter "<<j<<" of "<<distances.size()<<std::endl;
			if (distCounter.find(distances[j]) == distCounter.end())
				//distance[j] hasn't been counted before
				distCounter[distances[j]] = 0;
			distCounter[distances[j]] += 1;
		}

		std::vector<float> dists(distCounter.size()), weights(
				distCounter.size());

		for (const auto& n : distCounter) {
//			std::cout<<n.first <<" "<<n.second<<std::endl;
			dists.push_back((float) n.first + 1); // the key is the distance, which needs adjustment
			weights.push_back((float) n.second); //the value is the number of times it has been counted
		}

//		for (int k = 0; k < dists.size(); k++)
//			std::cout << dists[k] << " " << weights[k] << std::endl;
//        std::cout << dists.size() << " " << weights.size() << std::endl;

//		std::cout<<MathUtils::calculateMeanWithoutZeroes(weights)<<std::endl;
		features[i] = std::make_tuple(
				MathUtils::calculateWeightedAverage(dists,weights),
				MathUtils::calculateWeightedStd(dists,weights));

//		cout<<std::get<0>(features[i]) << " "<<std::get<1>(features[i])<<std::endl;
	}

	return features;
}

BfsMomentsCalculator::~BfsMomentsCalculator() {
}
