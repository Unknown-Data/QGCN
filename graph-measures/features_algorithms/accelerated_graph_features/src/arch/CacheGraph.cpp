/*
 * CacheGraph.cpp
 * 
 * A container for the cache aware representation of a graph.
 *
 *  Created on: Oct 28, 2018
 *      Author: ori
 */

#include "../includes/CacheGraph.h"

void CacheGraph::Clear() {
	m_NumberOfNodes = 0;
	m_NumberOfEdges = 0;
	delete m_Graph;
	m_Graph = NULL;
	delete m_Offsets;
	m_Offsets = NULL;
}

/*
 Initialize the graph from two lists (the offset list and the adjacency list).
 */
void CacheGraph::Assign(const std::vector<int64>& NodeOffsets,
		const std::vector<unsigned int>& Neighbours) {
	//Clear the graph so we can assign new data
	Clear();

	//copy the offset list into a new array
	m_NumberOfNodes = static_cast<unsigned int>(NodeOffsets.size() - 1);
	m_Offsets = new int64[m_NumberOfNodes + 1];
	std::memcpy(m_Offsets, &NodeOffsets[0], NodeOffsets.size() * sizeof(int64));

	m_NumberOfEdges = m_Offsets[m_NumberOfNodes];
	//copy the adjacency list into a new array
	m_Graph = new unsigned int[m_NumberOfEdges];
	std::memcpy(m_Graph, &Neighbours[0],
			Neighbours.size() * sizeof(unsigned int));
	/*	LOG(INFO) << m_NumberOfNodes << '\t' << m_Offsets[m_NumberOfNodes];
	 //std::cout << "Nodes: " << m_NumberOfNodes << " Edges:" << m_NumberOfEdges
	 << std::endl;*/

}

void CacheGraph::Assign(const std::vector<int64>& NodeOffsets,
		const std::vector<unsigned int>& Neighbours,
		const std::vector<double>& weights) {
	this->Assign(NodeOffsets, Neighbours);
	m_Weights = new double[m_NumberOfEdges];
	std::memcpy(m_Weights, &weights[0], weights.size() * sizeof(double));
	weighted = true;
}

/*
 Save the graph to a binary file.
 */
bool CacheGraph::SaveToFile(const std::string& FileName) const {

	//open or create a file of a binary format
	FILE* hFile = std::fopen(FileName.c_str(), "w+b");

	//write the class variables to the file
	std::fwrite(&m_NumberOfNodes, sizeof(unsigned int), 1, hFile);
	std::fwrite(&m_NumberOfEdges, sizeof(int64), 1, hFile);
	std::fwrite(m_Offsets, sizeof(int64), m_NumberOfNodes + 1, hFile);
	std::fwrite(m_Graph, sizeof(unsigned int), m_NumberOfEdges, hFile);
	std::fwrite(&weighted, sizeof(bool), 1, hFile);
	if (weighted)
		std::fwrite(m_Weights, sizeof(double), m_NumberOfEdges, hFile);
	std::fwrite(&directed, sizeof(bool), 1, hFile);

	//close the file
	std::fclose(hFile);
	hFile = NULL;

	//LOG(INFO) << m_NumberOfNodes << '\t' << m_Offsets[m_NumberOfNodes];
//	//std::cout << m_NumberOfNodes << '\t' << m_Offsets[m_NumberOfNodes];

	return true;

}

/*
 Read the data from the binary file
 */
bool CacheGraph::LoadFromFile(const std::string& FileName) {
	Clear();
	FILE* hFile;
	//if (!Utilities::FileExists(FileName)) { LOG(ERROR) << " Missing network file. Run CacheGraph::LoadFromFile - " << FileName; return false; }
	//open network file
	try {
		hFile = std::fopen(FileName.c_str(), "rb");
	} catch (std::exception& e) {
		/*LOG(ERROR) << " Missing network file. Run CacheGraph::LoadFromFile - " << FileName; return false;*/
//		//std::cout << " Missing network file. Run CacheGraph::LoadFromFile - "
//				<< FileName;
		return false;
	}

	//read the number of nodes
	std::fread(&m_NumberOfNodes, sizeof(unsigned int), 1, hFile);

	//read the number of edges
	int64 NumberOfEdges = 0;
	std::fread(&NumberOfEdges, sizeof(int64), 1, hFile);
	m_NumberOfEdges = NumberOfEdges;
	//create an array to store the indices (offsets) of the nodes in the graph array and read into it
	m_Offsets = new int64[m_NumberOfNodes + 1];
	std::fread(m_Offsets, sizeof(int64), m_NumberOfNodes + 1, hFile);

	//create the main array containing the lists of neighbors.
	/*
	 NOTE:
	 Here there is an assumption of a directed graph.
	 An undirected graph is created by having two edges saved in the file for each edge
	 in the network.
	 */
	m_Graph = new unsigned int[NumberOfEdges];
	std::fread(m_Graph, sizeof(unsigned int), NumberOfEdges, hFile);
	std::fread(&weighted, sizeof(bool), 1, hFile);
	if (weighted)
		std::fread(m_Weights, sizeof(double), m_NumberOfEdges, hFile);
	std::fread(&directed, sizeof(bool), 1, hFile);

	std::fclose(hFile);
	hFile = NULL;
	return true;

}

/*
 Utility function: create the full path from the directory and the file name and then call the
 overloaded function.
 */
bool CacheGraph::LoadFromFile(const std::string& DirectroyName,
		const std::string& BaseFileName) {
	std::string FileName = GetFileNameFromFolder(DirectroyName, BaseFileName);
	return LoadFromFile(FileName);
}

/*
 Utility function: append the directory and file names into one string
 */
std::string CacheGraph::GetFileNameFromFolder(const std::string& DirectroyName,
		const std::string& BaseFileName) {
	std::stringstream FileName;
	FileName << DirectroyName << BaseFileName << "_" << std::setfill('0')
			<< std::setw(2) << ".bin";
	return FileName.str();
}

/*
 Assuming our graph is directed, generate the inverted graph.
 i.e. change every edge e=(a,b) to (b,a) in the inverted graph.
 */
void CacheGraph::InverseGraph(CacheGraph& InvertedGraph) const {
	//get the number of edges in the graph
	const int64 NumberOfEdges = m_Offsets[m_NumberOfNodes];
	//clear the inverted graph
	InvertedGraph.Clear();
	//assign the member variables to the inverted graph
	InvertedGraph.m_NumberOfNodes = m_NumberOfNodes;
	//allocate the needed memory
	InvertedGraph.m_Offsets = new int64[m_NumberOfNodes + 1];
	InvertedGraph.m_Graph = new unsigned int[NumberOfEdges];
//	InvertedGraph.m_Weights = new double[NumberOfEdges]; // TODO M_WEIGHTS

	//invert the adjancency list
	unsigned int* InDegrees = new unsigned int[m_NumberOfNodes];
	std::memset(InDegrees, 0, m_NumberOfNodes * sizeof(unsigned int));
	for (const unsigned int* p = m_Graph; p < m_Graph + NumberOfEdges; ++p) {
		++InDegrees[*p];
	}
	InvertedGraph.m_Offsets[0] = 0;
	std::partial_sum(InDegrees, InDegrees + m_NumberOfNodes,
			InvertedGraph.m_Offsets + 1);
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
//		for (const unsigned int* peer = m_Graph + m_Offsets[NodeID];
//				peer < m_Graph + m_Offsets[NodeID + 1]; ++peer) {
		for (int peerIndex = m_Offsets[NodeID];
				peerIndex < m_Offsets[NodeID + 1]; peerIndex++) {
			unsigned int peer = m_Graph[peerIndex];
			InvertedGraph.m_Graph[InvertedGraph.m_Offsets[peer]] = NodeID;
//			InvertedGraph.m_Weights[InvertedGraph.m_Offsets[peer]] = m_Weights[peerIndex];  // TODO M_WEIGHTS
			++InvertedGraph.m_Offsets[peer];
		}
	}
	InvertedGraph.m_Offsets[0] = 0;
	std::partial_sum(InDegrees, InDegrees + m_NumberOfNodes,
			InvertedGraph.m_Offsets + 1);
	//clear memory
	delete[] InDegrees;
	InDegrees = NULL;

}

/*
 * Create the undirected version of the graph represented in this instance.
 * Essentially, we're combining the directed graph and it's inverse.
 */
void CacheGraph::CureateUndirectedGraph(const CacheGraph& InvertedGraph,
		CacheGraph& UndirectedGraph) const {
	//calculate the number of edges in the graph
	int64 NumberOfEdges = m_Offsets[m_NumberOfNodes];
	//clear the new graph
	UndirectedGraph.Clear();
	//assign member variables
	UndirectedGraph.m_NumberOfNodes = m_NumberOfNodes;
	//allocate memory
	UndirectedGraph.m_Offsets = new int64[m_NumberOfNodes + 1];
	std::memset(UndirectedGraph.m_Offsets, 0,
			(m_NumberOfNodes) * sizeof(int64));
	//note that in an undirected graph, there are twice as many edges in the
	//adjacency list as in a directed graph.
	unsigned int* temp_Graph = new unsigned int[2 * NumberOfEdges];
//	double* temp_weights = new double[2 * NumberOfEdges];  // TODO M_WEIGHTS
	std::memset(temp_Graph, 0, 2 * NumberOfEdges * sizeof(unsigned int));
	unsigned int *temp_graph_pointer = temp_Graph;
//	double* temp_weight_pointer = temp_weights;     // TODO M_WEIGHTS

	//combine the inverted and normal graphs into an undirected one.
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
		auto p1 = m_Graph + m_Offsets[NodeID]; //current neighbor
		auto p2 = InvertedGraph.m_Graph + InvertedGraph.m_Offsets[NodeID]; //inverted current neighbor

//		auto w1 = m_Weights + m_Offsets[NodeID]; //current weight  // TODO M_WEIGHTS
//		auto w2 = InvertedGraph.m_Weights + InvertedGraph.m_Offsets[NodeID]; //inverted current weight // TODO M_WEIGHTS

		while (p1 < m_Graph + m_Offsets[NodeID + 1]
				&& p2
						< InvertedGraph.m_Graph
								+ InvertedGraph.m_Offsets[NodeID + 1]) { //while we are in both neighbor lists
			if (*p1 == *p2) { //bi-directional edge
				*temp_graph_pointer = *p1;
//				*temp_weight_pointer = *w1;  // TODO M_WEIGHTS
				++p1;
				++p2;
//				++w1;                        // TODO M_WEIGHTS
//				++w2;                        // TODO M_WEIGHTS
			} else if (*p1 < *p2) {
				*temp_graph_pointer = *p1;
//				*temp_weight_pointer = *w1;  // TODO M_WEIGHTS
				++p1;
//				++w1;                        // TODO M_WEIGHTS
			} else // if (*p1>*p2)
			{
				*temp_graph_pointer = *p2;
//				*temp_weight_pointer = *w2;  // TODO M_WEIGHTS
				++p2;
//				++w2;                        // TODO M_WEIGHTS
			}
			++temp_graph_pointer;
		} //END WHILE
		if (p1 < m_Graph + m_Offsets[NodeID + 1]) { //if we haven't read all of the neighbors of the node
			int64 RemainingElements = m_Graph + m_Offsets[NodeID + 1] - p1;
			std::memcpy(temp_graph_pointer, p1,
					sizeof(unsigned int) * RemainingElements);
//			std::memcpy(temp_weight_pointer, p1,
//					sizeof(double) * RemainingElements);     // TODO M_WEIGHTS
			temp_graph_pointer += RemainingElements;
//			temp_weight_pointer += RemainingElements;        // TODO M_WEIGHTS
		} else if (p2
				< InvertedGraph.m_Graph + InvertedGraph.m_Offsets[NodeID + 1]) { //or if we haven't read all the neighbors in the inverted graph
			int64 RemainingElements = InvertedGraph.m_Graph
					+ InvertedGraph.m_Offsets[NodeID + 1] - p2;
			std::memcpy(temp_graph_pointer, p2,
					sizeof(unsigned int) * RemainingElements);
//			std::memcpy(temp_weight_pointer, p2,
//					sizeof(double) * RemainingElements);      // TODO M_WEIGHTS
			temp_graph_pointer += RemainingElements;
//			temp_weight_pointer += RemainingElements;         // TODO M_WEIGHTS

		}
		//Otherwise - we've read all neighbors

		UndirectedGraph.m_Offsets[NodeID + 1] = temp_graph_pointer - temp_Graph;
	}

	UndirectedGraph.m_NumberOfEdges =
			UndirectedGraph.m_Offsets[m_NumberOfNodes];

	// Copy neighbor list from temp
	UndirectedGraph.m_Graph = new unsigned int[UndirectedGraph.m_NumberOfEdges];
	std::memcpy(UndirectedGraph.m_Graph, temp_Graph,
			UndirectedGraph.m_NumberOfEdges * sizeof(unsigned int));

	//Copy weights from temp
	UndirectedGraph.m_Weights = new double[UndirectedGraph.m_NumberOfEdges];
//	std::memcpy(UndirectedGraph.m_Weights,temp_weights,UndirectedGraph.m_NumberOfEdges*sizeof(double));  // TODO M_WEIGHTS
	//clear memory
	delete[] temp_Graph;
}

std::vector<unsigned int> CacheGraph::ComputeNodeDegrees() const {
	std::vector<unsigned int> Degrees(m_NumberOfNodes, 0);
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
		Degrees[NodeID] = static_cast<unsigned int>(m_Offsets[NodeID + 1]
				- m_Offsets[NodeID]);
	}
	return Degrees;
}

std::vector<float> CacheGraph::ComputeNodePageRank(float dumping,
		unsigned int NumberOfIterations) const {

	std::random_device rng;
	std::mt19937 urng(rng());
	std::vector<float> PageRank(m_NumberOfNodes, 1.0F - dumping);
	while (NumberOfIterations > 0) {
		std::vector<int> NodeIDs(m_NumberOfNodes); // vector with 100 ints.
		std::iota(std::begin(NodeIDs), std::end(NodeIDs), 0);
		std::shuffle(NodeIDs.begin(), NodeIDs.end(), urng);
		for (unsigned int index = 0; index < m_NumberOfNodes; ++index) {
			const unsigned int NodeID = NodeIDs[index];
			float friends_contribution = 0.0;
			for (auto p = m_Graph + m_Offsets[NodeID];
					p < m_Graph + m_Offsets[NodeID + 1]; ++p) {
				friends_contribution += PageRank[*p]
						/ (m_Offsets[*p + 1] - m_Offsets[*p]);
			}
			PageRank[NodeID] = (1.0F - dumping)
					+ dumping * friends_contribution;
		}
		--NumberOfIterations;
	}
	return PageRank;
}

std::vector<unsigned short> CacheGraph::ComputeKCore() const {
	const unsigned short UNSET_K_CORE = static_cast<unsigned short>(-1);
	std::vector<unsigned short> KShell(m_NumberOfNodes, UNSET_K_CORE);
	std::vector<unsigned int> Degrees(m_NumberOfNodes, 0);
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
		Degrees[NodeID] = static_cast<unsigned int>(m_Offsets[NodeID + 1]
				- m_Offsets[NodeID]);
		if (Degrees[NodeID] == 0) {
			KShell[NodeID] = 0;
		}
	}
	unsigned short CurrentShell = 1;
	bool NodesInShell = false;
	do {
		NodesInShell = false;
		bool any_degree_changed = false;
		do {
			any_degree_changed = false;
			for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {

				if (KShell[NodeID] == UNSET_K_CORE
						&& Degrees[NodeID] <= CurrentShell) {
					KShell[NodeID] = CurrentShell;
					NodesInShell = true;
					for (auto p = m_Graph + m_Offsets[NodeID];
							p < m_Graph + m_Offsets[NodeID + 1]; ++p) {
						if (KShell[*p] == UNSET_K_CORE) {
							--Degrees[*p];
							any_degree_changed = true;
						}
					}
				}
			}
		} while (any_degree_changed);
		++CurrentShell;
	} while (NodesInShell);
	return KShell;
}

/*
 Check wether q is a neighbor of p (i.e. if there exists an edge p -> q)
 For an undirected graph, the order does not matter.
 Input: the two nodes to check
 Output: whether there is an edge p->q
 Note: we are working under the assumption that the list of p's neighbors is ordered,
 and so we use binary search.
 Usage of the binary search keeps the proccess to O(log(V))
 */
bool CacheGraph::areNeighbors(const unsigned int p,
		const unsigned int q) const {
	int first = m_Offsets[p],  //first array element
			last = m_Offsets[p + 1] - 1,     //last array element
			middle;                       //mid point of search
//	std::cout << "In bin search"<<std::endl;
//	std::cout << "p="<<p<<" q="<<q<<std::endl;
	while (first <= last) {
		middle = (int) (first + last) / 2; //this finds the mid point
//		std::cout << "Binary search: " << first<< " "<< last<< " "<< middle << std::endl;
		//TODO: fix overflow problem
		if (m_Graph[middle] == q) {
			return true;
		} else if (m_Graph[middle] < q)
				{
			first = middle + 1;     //if it's in the upper half

		} else { //m_Graph[middle] < q
			last = middle - 1; // if it's in the lower half
		}
	}
	return false;  // not found

}

std::vector<unsigned int>* CacheGraph::SortedNodesByDegree() const {
	std::vector<unsigned int>* sortedNodes = new std::vector<unsigned int>();
	sortedNodes->reserve(m_NumberOfNodes);

	std::vector<unsigned int> nodeDegrees = ComputeNodeDegrees();
	std::vector<NodeWithDegree> nodesWithDegrees;
	nodesWithDegrees.reserve(m_NumberOfNodes);
	for (unsigned int node = 0; node < m_NumberOfNodes; node++)
		nodesWithDegrees.push_back( { node, nodeDegrees[node] });

	std::sort(nodesWithDegrees.begin(), nodesWithDegrees.end(),
			cmpNodesByDegree);

	for (NodeWithDegree nd : nodesWithDegrees)
		sortedNodes->push_back(nd.node);
	return sortedNodes;
}
