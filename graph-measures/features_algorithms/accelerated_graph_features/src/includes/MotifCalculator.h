/*
 * MotifCalculator.h
 *
 *  Created on: Dec 2, 2018
 *      Author: ori
 */

#ifndef FEATURES_MOTIFCALCULATOR_H_
#define FEATURES_MOTIFCALCULATOR_H_

#include "FeatureCalculator.h"
#include "MotifUtils.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <set>

/**
 * The motif calc returns a list for each node counting the motifs in it.
 */
class MotifCalculator: public FeatureCalculator<vector<vector<unsigned int>*>*> {
public:
	MotifCalculator(int level,bool directed);
	MotifCalculator(int level,bool directed, string motif_path);
	virtual vector<vector<unsigned int>*>* Calculate();
	virtual ~MotifCalculator();

private:
	void Motif3Subtree(unsigned int node);
	void Motif4Subtree(unsigned int node);

	void InitFeatureCounters();
	void LoadMotifVariations(int level,bool directed);
	void SetAllMotifs();
	void SetSortedNodes();
	void SetRemovalIndex();
	virtual void init();
	void GroupUpdater(std::vector<unsigned int> group);
	int GetGroupNumber(std::vector<unsigned int> group);

    string MOTIF_VARIATIONS_PATH;
    CacheGraph fullGraph;

	//Either 3 or 4
	unsigned int level;
	//The CacheGraph is always directed, so we need to specify the motif variation
	bool directed;
	//map the group num to the iso motif
	std::map<unsigned int,int>* nodeVariations;

	//list of base motifs
	std::vector<int>* allMotifs;
	//the index in which we remove the node from the graph. Basically, from this index on the node doesen't exist.
	std::vector<unsigned int>* removalIndex;
	//the nodes, sorted in descending order by the degree.
	std::vector<unsigned int>* sortedNodesByDegree;

	//the results, node -> {motif-> motif_count}
	vector<vector<unsigned int>*>* features;


};

#endif /* FEATURES_MOTIFCALCULATOR_H_ */
