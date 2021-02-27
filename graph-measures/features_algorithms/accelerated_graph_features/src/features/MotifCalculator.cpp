/*
 * MotifCalculator.cpp
 *
 *  Created on: Dec 2, 2018
 *      Author: ori
 */

#include "../includes/MotifCalculator.h"
#include "../includes/MotifVariationConstants.h"
#include <algorithm>
void MotifCalculator::init() {
	CacheGraph inverse(true);
	mGraph->InverseGraph(inverse);
	mGraph->CureateUndirectedGraph(inverse, fullGraph);

	//std::cout << "Load variations" << std::endl;
	this->LoadMotifVariations(level, directed);
	//std::cout << "All motifs" << std::endl;
	this->SetAllMotifs();
	//std::cout << "Sorted Nodes" << std::endl;
	this->SetSortedNodes();
	//std::cout << "Removal Index" << std::endl;
	this->SetRemovalIndex();
	//std::cout << "Feature counters" << std::endl;
	this->InitFeatureCounters();
	std::cout << "Done" << std::endl;
	//std::cout << this->removalIndex->size() << std::endl;

}

MotifCalculator::MotifCalculator(int level, bool directed, string motif_path) :
		directed(directed), nodeVariations(NULL), allMotifs(NULL), removalIndex(
		NULL), sortedNodesByDegree(NULL), fullGraph(false) {
	MOTIF_VARIATIONS_PATH = motif_path;
	//check level
	if (level != 3 && level != 4)
		throw invalid_argument("Level must be 3 or 4");
	this->level = level;
	this->features = new std::vector<vector<unsigned int>*>;
//	interesting_motifs =  {3, 4, 6, 15, 17,23, 24, 26, 27, 29, 78, 80};

}

void MotifCalculator::InitFeatureCounters() {
	for (int node = 0; node < mGraph->GetNumberOfNodes(); node++) {
		vector<unsigned int>* motifCounter = new vector<unsigned int>;
		std::set<unsigned int> s(this->allMotifs->begin(),
				this->allMotifs->end());
		for (auto motif : s)
			if (motif != -1)
//				(*motifCounter)[motif] = 0;
				motifCounter->push_back(0);

		features->push_back(motifCounter);
	}
	delete this->allMotifs;
}

void MotifCalculator::LoadMotifVariations(int level, bool directed) {

	this->nodeVariations = new std::map<unsigned int, int>();
//
//	string suffix;
//	if (directed)
//		suffix = "_directed_cpp.txt";
//	else
//		suffix = "_undirected_cpp.txt";
//
//	string fileName = MOTIF_VARIATIONS_PATH + "/" + std::to_string(level)
//			+ suffix;
//	std::ifstream infile(fileName);
	const char* motifVariations[4] = { undirected3, directed3, undirected4,
			directed4 };

	int variationIndex = 2 * (level - 3) + (directed ? 1 : 0);
	std::istringstream f(motifVariations[variationIndex]);
	std::string line;
	std::string a, b;
	while (getline(f, line)) {
		int x, y;
		int n = line.find(" ");
		a = line.substr(0, n);
		b = line.substr(n);
		try {
			x = stoi(a);
			y = stoi(b);
		} catch (exception &e) {
			y = -1;

		}
//		cout << line << endl;
//		cout << x << ":" << y << endl;

		(*nodeVariations)[x] = y;
	}

}

MotifCalculator::MotifCalculator(int level, bool directed) :
		MotifCalculator(level, directed, "../src/features/motif_variations") {
}

void MotifCalculator::SetAllMotifs() {
	this->allMotifs = new std::vector<int>();

	for (const auto& x : *(this->nodeVariations))
		this->allMotifs->push_back(x.second);
}

void MotifCalculator::SetSortedNodes() {
	this->sortedNodesByDegree = mGraph->SortedNodesByDegree();
}
/**
 * We iterate over the list of sorted nodes.
 * If node p is in index i in the list, it means that it will be removed after the i-th iteration,
 * or conversely that it is considered "not in the graph" from iteration i+1 onwards.
 * This means that any node with a removal index of j is not considered from iteration j+1.
 * (The check is (removalIndex[node] > currentIteration).
 */
void MotifCalculator::SetRemovalIndex() {
	this->removalIndex = new std::vector<unsigned int>();
	for (int i = 0; i < mGraph->GetNumberOfNodes(); i++) {
		removalIndex->push_back(0);
	}
	for (unsigned int index = 0; index < mGraph->GetNumberOfNodes(); index++) {
		auto node = sortedNodesByDegree->at(index);
		removalIndex->at(node) = index;
	}
}

vector<vector<unsigned int>*>* MotifCalculator::Calculate() {

	if (this->level == 3) {
		//std::cout << "Start 3" << std::endl;

		for (auto node : *(this->sortedNodesByDegree)) {
			//std::cout << node << std::endl;
			Motif3Subtree(node);
		}
	} else {
		//std::cout << "Start 4" << std::endl;

		for (auto node : *(this->sortedNodesByDegree))
			Motif4Subtree(node);
	}
//	std::cout << "Done All" << std::endl;
	return this->features;
}

void MotifCalculator::Motif3Subtree(unsigned int root) {
	// Instead of yield call GroupUpdater function
	// Don't forget to check each time that the nodes are in the graph (check removal index).
	int idx_root = this->removalIndex->at(root);// root_idx is also our current iteration -
	std::map<unsigned int, int> visited_vertices;// every node_idx smaller than root_idx is already handled
	visited_vertices[root] = 0;
	int visit_idx = 1;

	const unsigned int* neighbors = fullGraph.GetNeighborList();// all neighbors - ancestors and descendants
	const int64* offsets = fullGraph.GetOffsetList();

	// TODO problem with dual edges
	//std::cout << "Mark" << std::endl;
	for (int64 i = offsets[root]; i < offsets[root + 1]; i++) // loop first neighbors
		if (this->removalIndex->at(neighbors[i]) > idx_root) // n1 not handled yet
			visited_vertices[neighbors[i]] = visit_idx++;
	//std::cout << "Mark" << std::endl;
	for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
		unsigned int n1 = neighbors[n1_idx];
		if (this->removalIndex->at(n1) <= idx_root)		// n1 already handled
			continue;
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) {// loop second neighbors
			unsigned int n2 = neighbors[n2_idx];
			if (this->removalIndex->at(n2) <= idx_root)	// n2 already handled
				continue;
			if (visited_vertices.find(n2) != visited_vertices.end()) { // check if n2 was visited &&
				if (visited_vertices[n1] < visited_vertices[n2]) // n2 discovered after n1  TODO VERIFY
					this->GroupUpdater(
							std::vector<unsigned int> { root, n1, n2 }); // update motif counter [r,n1,n2]
			} else {
				visited_vertices[n2] = visit_idx++;

				this->GroupUpdater(std::vector<unsigned int> { root, n1, n2 }); // update motif counter [r,n1,n2]
			}	// end ELSE
		}	// end LOOP_SECOND_NEIGHBORS

	}	// end LOOP_FIRST_NEIGHBORS
	//std::cout << "Mark" << std::endl;
	vector<vector<unsigned int> *> *n1_comb = neighbors_combinations(neighbors,
			offsets[root], offsets[root + 1]);
	for (auto it = n1_comb->begin(); it != n1_comb->end(); ++it) {
		unsigned int n1 = (**it)[0];
		unsigned int n2 = (**it)[1];
		//std::cout << "\t" << n1 << "," << n2 << std::endl;
		if (this->removalIndex->at(n1) <= idx_root
				|| this->removalIndex->at(n2) <= idx_root)// motif already handled
			continue;
		//std::cout << "Mark1" << std::endl;
		//std::cout << (visited_vertices[n1] < visited_vertices[n2]) << std::endl;
		//std::cout << mGraph->areNeighbors(n1, n2) << std::endl;
		//std::cout << mGraph->areNeighbors(n2, n1) << std::endl;
		if ((visited_vertices[n1] < visited_vertices[n2])
				&& !(mGraph->areNeighbors(n1, n2)
						|| mGraph->areNeighbors(n2, n1)))// check n1, n2 not neighbors
			//std::cout << "Mark2" << std::endl;
			this->GroupUpdater(std::vector<unsigned int> { root, n1, n2 });	// update motif counter [r,n1,n2]
	}	// end loop COMBINATIONS_NEIGHBORS_N1
	for (auto p : *n1_comb) {
		delete p;
	}
	delete n1_comb;
}

void MotifCalculator::Motif4Subtree(unsigned int root) {
	int idx_root = this->removalIndex->at(root);// root_idx is also our current iteration -
	std::map<unsigned int, int> visited_vertices;// every node_idx smaller than root_idx is already handled
	visited_vertices[root] = 0;

	const unsigned int* neighbors = fullGraph.GetNeighborList();// all neighbors - ancestors and descendants
	const int64* offsets = fullGraph.GetOffsetList();

	// TODO problem with dual edges
	for (int64 i = offsets[root]; i < offsets[root + 1]; i++) // loop first neighbors
		if (this->removalIndex->at(neighbors[i]) > idx_root) // n1 not handled yet
			visited_vertices[neighbors[i]] = 1;

	// TODO: add combinations_3 for first neighbors
	/*
	 *    for n1, n2, n3 in combinations(neighbors_first_deg, 3):
	 yield [root, n1, n2, n3]
	 */
	int64 end = offsets[root + 1];
	for (int64 i = offsets[root]; i < end; i++) {
		for (int64 j = i + 1; j < end; j++) {
			if (j == end - 1) //if j is the last element, we can't add an element and therefore it's not a 3-combination
				continue;
			for (int64 k = j + 1; k < end; k++) {
				unsigned int n11 = neighbors[i];
				unsigned int n12 = neighbors[j];
				unsigned int n13 = neighbors[k];
				if (this->removalIndex->at(n11) <= idx_root
						|| this->removalIndex->at(n12) <= idx_root
						|| this->removalIndex->at(n13) <= idx_root) // motif already handled
					continue;
				this->GroupUpdater(std::vector<unsigned int> { root, n11, n12,
						n13 }); // update motif counter [r,n11,n12,n13]
			}
		}
	}

	// All other casesDone All
	for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
		unsigned int n1 = neighbors[n1_idx];
		if (this->removalIndex->at(n1) <= idx_root)	// n1 already handled
			continue;
		//Mark second neighbors
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) {// loop second neighbors
			unsigned int n2 = neighbors[n2_idx];
			if (this->removalIndex->at(n2) <= idx_root)	// n2 already handled
				continue;
			if (visited_vertices.find(n2) == visited_vertices.end()) { // check if n2 was *not* visited
				visited_vertices[n2] = 2;

			} //end if
		} //end loop SECOND NEIGHBORS

		// The case of root-n1-n2-n11
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (again)
			unsigned int n2 = neighbors[n2_idx];
			if (this->removalIndex->at(n2) <= idx_root)	// n2 already handled
				continue;
			for (int64 n11_idx = offsets[root]; n11_idx < offsets[root + 1];
					n11_idx++) { // loop first neighbors
				unsigned int n11 = neighbors[n11_idx];
				if (this->removalIndex->at(n11) <= idx_root) // n2 already handled
					continue;
				if (visited_vertices[n2] == 2 && n11 != n1) { //TODO: verify
					bool edgeExists = mGraph->areNeighbors(n2, n11)
							|| mGraph->areNeighbors(n11, n2);
					if (!edgeExists || (edgeExists && n1 < n11)) {
						this->GroupUpdater(std::vector<unsigned int> { root, n1,
								n11, n2 }); // update motif counter [r,n1,n11,n2]
					} // end if
				}
			} // end loop INNER FIRST NEIGHBORS
		} // end loop SECOND NEIGHBORS AGAIN

		// The case of root-n1-n21-n22
		//2-combinations on second neighbors
		end = offsets[n1 + 1];
		for (int64 i = offsets[n1]; i < end; i++) {
			for (int64 j = i + 1; j < end; j++) {

				unsigned int n21 = neighbors[i];
				unsigned int n22 = neighbors[j];
				if (this->removalIndex->at(n21) <= idx_root
						|| this->removalIndex->at(n22) <= idx_root) // motif already handled
					continue;
				if (2 == visited_vertices[n21] && visited_vertices[n22] == 2) {
					this->GroupUpdater(std::vector<unsigned int> { root, n1,
							n21, n22 }); // update motif counter [r,n1,n21,n22]
				}
			}
		} // end loop SECOND NEIGHBOR COMBINATIONS

	}
	//The case of n1-n2-n3
	for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
		unsigned int n1 = neighbors[n1_idx];
		if (this->removalIndex->at(n1) <= idx_root)	// n1 already handled
			continue;
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (third time's the charm)
			unsigned int n2 = neighbors[n2_idx];
			if (this->removalIndex->at(n2) <= idx_root)	// n2 already handled
				continue;
			if (visited_vertices[n2] == 1)
				continue;

			for (int64 n3_idx = offsets[n2]; n3_idx < offsets[n2 + 1];
					n3_idx++) { // loop third neighbors
				unsigned int n3 = neighbors[n3_idx];
				if (this->removalIndex->at(n3) <= idx_root)	// n2 already handled
					continue;


				if (visited_vertices.find(n3) == visited_vertices.end()) { // check if n3 was *not* visited
					visited_vertices[n3] = 3;
					if (visited_vertices[n2] == 2) { // check if n2 is a visited second neighbor
						this->GroupUpdater(std::vector<unsigned int> { root, n1,
								n2, n3 }); // update motif counter [r,n1,n2,n3]
					} // end check if n2 is a visited second neighbor
				} // end check if n3 was not visited

				else {
					if (visited_vertices[n3] == 1)
						continue;


					if (visited_vertices[n3] == 2
							&& !(mGraph->areNeighbors(n1, n3)
									|| mGraph->areNeighbors(n3, n1))) {
						this->GroupUpdater(std::vector<unsigned int> { root, n1,
								n2, n3 }); // update motif counter [r,n1,n2,n3]
					} else if (visited_vertices[n3] == 3
							&& visited_vertices[n2] == 2) { //TODO: verify
						this->GroupUpdater(std::vector<unsigned int> { root, n1,
								n2, n3 }); // update motif counter [r,n1,n2,n3]
					} // end if

				} //end else
			} // end loop THIRD NEIGHBORS
		} // end loop SECOND NEIGHBORS THIRD TIME'S THE CHARM

	} // end loop FIRST NEIGHBORS

}

void MotifCalculator::GroupUpdater(std::vector<unsigned int> group) {
// TODO: count overall number of motifs in graph (maybe different class)?
	int groupNumber = GetGroupNumber(group);
	int motifNumber = (*(this->nodeVariations))[groupNumber];

//	int interestingMotifs[] = { 3, 4, 6, 15, 17, 23, 24, 26, 27, 29, 78, 80 };

//	if(std::find(group.begin(),group.end(),0)!=group.end() && motifNumber == 17)
//		std::cout << "A 0/17 group "<<group[0]<<","<<group[1]<<","<<group[2]<<","<<group[3]<<std::endl;
//	if (std::find(group.begin(),group.end(),4)!=group.end()) {
//
//		if (motifNumber == 15) {
//			std::cout << motifNumber << " : ";
//			for (auto n : group)
//				std::cout << n << ",";
//			std::cout << std::endl;
//		}
//	}
//	std::cout << motifNumber << ",";
//	for (auto n : group)
//		std::cout << n << ",";
//	std::cout << std::endl;
	if (motifNumber != -1)
		for (auto node : group)
			(*(*features)[node])[motifNumber]++;

}

int MotifCalculator::GetGroupNumber(std::vector<unsigned int> group) {
	vector<vector<unsigned int> *> * options;
	unsigned int n1, n2;

	vector<bool> edges;

	if (directed) {
		options = permutations(group);

	} else {
		options = combinations(group);
	}

	for (int i = 0; i < options->size(); i++) {
		n1 = options->at(i)->at(0);
		n2 = options->at(i)->at(1);
		edges.push_back(this->mGraph->areNeighbors(n1, n2));
	}

	for (auto p : *options) {
		delete p;
	}
	delete options;
	return bool_vector_to_int(edges);

}

MotifCalculator::~MotifCalculator() {
	//map the group num to the iso motif
	delete nodeVariations;
	//the index in which we remove the node from the graph. Basically, from this index on the node doesen't exist.
	delete removalIndex;
	//the nodes, sorted in descending order by the degree.
	delete sortedNodesByDegree;

}
