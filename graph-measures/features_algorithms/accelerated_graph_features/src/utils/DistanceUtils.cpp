#include "../includes/DistanceUtils.h"




/*
   Calculate the distance of all nodes from a single node (the origin) using BFS.

Assumptions:
1) The graph is not weighted. For weighted graphs consider using single_source_dijkstra.
2) There are no loop in the graph
Input:
g - a constant pointer to the GraphSnapsot to perform the calculation for
src - the integer ID of the origin node.
Output:
A vector that contains the distances of all the nodes from the origin.
The distance of node i is in position i in the vector.

The code is based on the code found here:https://www.geeksforgeeks.org/shortest-path-unweighted-graph/
*/
std::vector<unsigned int> DistanceUtils::BfsSingleSourceShortestPath(const CacheGraph * g,unsigned int src)
{
	const unsigned int numOfNodes = g->GetNumberOfNodes();

	std::vector<unsigned int> dist(numOfNodes);

	// a queue to maintain queue of vertices whose
	// adjacency list is to be scanned as per normal
	// DFS algorithm
	std::list<unsigned int> queue;

	// boolean array visited[] which stores the
	// information whether ith vertex is reached
	// at least once in the Breadth first search
	std::vector<bool> visited(numOfNodes);

	// initially all vertices are unvisited
	// so v[i] for all i is false
	// and as no path is yet constructed
	// dist[i] for all i set to infinity
	for (int i = 0; i < numOfNodes; i++) {
		visited[i] = false;
		dist[i] = INT_MAX;
	}

	// now source is first to be visited and
	// distance from source to itself should be 0
	visited[src] = true;
	dist[src] = 0;
	queue.push_back(src);

	//Get the neighbors
	const unsigned int* neighborList = g->GetNeighborList();
	const int64* offsetList = g->GetOffsetList();

	// standard BFS algorithm
	while (!queue.empty()) {
		int u = queue.front();
		queue.pop_front();
		//	for (int i = 0; i < adj[u].size(); i++) {
		//		if (visited[adj[u][i]] == false) {
		//			visited[adj[u][i]] = true;
		//			dist[adj[u][i]] = dist[u] + 1;
		//			pred[adj[u][i]] = u;
		//			queue.push_back(adj[u][i]);

		//			// We stop BFS when we find
		//			// destination.
		//			if (adj[u][i] == dest)
		//				return true;
		//		}
		//	}
		//}

		int64 begin_offset, end_offset;
		begin_offset = offsetList[u];
		end_offset = offsetList[u + 1];

		for (auto p = neighborList + begin_offset; p < neighborList + end_offset; ++p)
		{
			//p is a pointer to the nodes in the adjacency list.
			//Example usage: the offset of the current node is offsetList[*p]
			//In this case, p iterates over the neighbors of u
			if (!visited[*p]) {
				visited[*p] = true;
				dist[*p] = dist[u] + 1;
				queue.push_back(*p);
			}

		}

}

for (int i = 0; i < numOfNodes; i++) {
	if(dist[i] == INT_MAX)
		dist[i] = 0;
}



return dist;
}


/*
   Calculate the distance of all nodes from a single source using Dijkstra's algorithm.
Assumptions:
1) The graph has no negative weights

Input:
g - a constant pointer to the GraphSnapsot to perfom the calculation for
src - the integer ID of the origin node.

Output:
A vector that contains the distances of all the nodes from the origin.
The distance of node i is in position i in the vector.

The code is based on the code found here:https://www.geeksforgeeks.org/greedy-algorithms-set-6-dijkstras-shortest-path-algorithm/

*/
std::vector<float> DistanceUtils::DijkstraSingleSourceShortestPath(const CacheGraph * g, int src)
{
	const int numOfNodes = g->GetNumberOfNodes();

	//dist contains the shortest distance from src to other nodes
	std::vector<float> dist(numOfNodes);

	//sptSet[i] will be true if the distance to i is finalized
	std::vector<bool> sptSet(numOfNodes);

	//Init all distanced to INFINITY and sptSet to false
	for (int i = 0; i < numOfNodes; i++)
		dist[i] = INT_MAX, sptSet[i] = false;

	//use the dijkstra algorithm to calculate the minimun distances

	FibQueue<float> priorityQueue;



	return dist;
}

