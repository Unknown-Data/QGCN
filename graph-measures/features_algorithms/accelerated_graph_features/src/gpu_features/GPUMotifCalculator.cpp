/*
 * GPUMotifCalculator.cpp
 *
 *  Created on: Dec 2, 2018
 *      Author: ori
 */

#include "../includes/GPUMotifCalculator.h"
#include "../includes/MotifVariationConstants.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Global managed variables   ////////////////////////////////////////////////
__managed__ unsigned int globalNumOfNodes;
__managed__ unsigned int globalNumOfMotifs;
__managed__ unsigned int globalNumOfEdges;

__managed__ bool globalDirected;

// DEVICE VARIABLES

//Pointers to the device vectors declared above - no need to delete as the device_vectors live on the stack
__managed__ unsigned int* globalDevicePointerMotifVariations;
__managed__ unsigned int* globalDevicePointerRemovalIndex;
__managed__ unsigned int* globalDevicePointerSortedNodesByDegree;

// For the original graph
__managed__ int64* globalDeviceOriginalGraphOffsets;
__managed__ unsigned int* globalDeviceOriginalGraphNeighbors;

// For the full graph
__managed__ int64* globalDeviceFullGraphOffsets;
__managed__ unsigned int* globalDeviceFullGraphNeighbors;

// Feature array
__managed__ unsigned int* globalDeviceFeatures;

/////////////////////////////   END Global managed variables   /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUMotifCalculator::init() {
	CacheGraph inverse(true);
	mGraph->InverseGraph(inverse);
	mGraph->CureateUndirectedGraph(inverse, fullGraph);
	this->numOfNodes = this->mGraph->GetNumberOfNodes();
	this->numOfEdges = this->mGraph->GetNumberOfEdges();
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
	//std::cout << "Copy to GPU" << std::endl;
	this->CopyAllToDevice();
	//std::cout << "Done" << std::endl;
	//std::cout << this->removalIndex->size() << std::endl;
}

GPUMotifCalculator::GPUMotifCalculator(int level, bool directed) :
		directed(directed), nodeVariations(NULL), allMotifs(NULL), removalIndex(
		NULL), sortedNodesByDegree(NULL), fullGraph(false), numOfMotifs(0), deviceFeatures(
		NULL) {
	//check level
	if (level != 3 && level != 4)
		throw invalid_argument("Level must be 3 or 4");
	this->level = level;
	this->features = new std::vector<vector<unsigned int> *>;
}

void GPUMotifCalculator::InitFeatureCounters() {
	for (int node = 0; node < numOfNodes; node++) {
		vector<unsigned int> *motifCounter = new vector<unsigned int>;
		std::set<int> s(this->allMotifs->begin(), this->allMotifs->end());
		this->numOfMotifs = s.size() - 1;
		for (auto motif : s)
			if (motif != -1)
				//				(*motifCounter)[motif] = 0;
				motifCounter->push_back(0);

		features->push_back(motifCounter);
	}
	delete this->allMotifs;
}

void GPUMotifCalculator::LoadMotifVariations(int level, bool directed) {

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
	const int numOfMotifsOptions[4] = { 8, 64, 64, 4096 };

	int variationIndex = 2 * (level - 3) + (directed ? 1 : 0);
	this->nodeVariations = new std::vector<unsigned int>(
			numOfMotifsOptions[variationIndex]);
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

void GPUMotifCalculator::SetAllMotifs() {
	this->allMotifs = new std::vector<int>();

	for (const auto &x : *(this->nodeVariations))
		this->allMotifs->push_back(x);
}

void GPUMotifCalculator::SetSortedNodes() {
	this->sortedNodesByDegree = mGraph->SortedNodesByDegree();
}
/**
 * We iterate over the list of sorted nodes.
 * If node p is in index i in the list, it means that it will be removed after the i-th iteration,
 * or conversely that it is considered "not in the graph" from iteration i+1 onwards.
 * This means that any node with a removal index of j is not considered from iteration j+1.
 * (The check is (removalIndex[node] > currentIteration).
 */
void GPUMotifCalculator::SetRemovalIndex() {
	this->removalIndex = new std::vector<unsigned int>();
	for (int i = 0; i < numOfNodes; i++) {
		removalIndex->push_back(0);
	}
	for (unsigned int index = 0; index < numOfNodes; index++) {
		auto node = sortedNodesByDegree->at(index);
		removalIndex->at(node) = index;
	}
}

void GPUMotifCalculator::CopyAllToDevice() {
	//	thrust::device_vector<unsigned int> deviceMotifVariations; // @suppress("Type cannot be resolved")// @suppress("Symbol is not resolved")
	//	thrust::device_vector<unsigned int> deviceRemovalIndex; // @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")
	//	thrust::device_vector<unsigned int> deviceSortedNodesByDegree;// @suppress("Type cannot be resolved") // @suppress("Symbol is not resolved")

	/*
	 * 1) Allocate unified memory
	 * 2) Copy vectors to the memory
	 * 3) delete the memory in d'tor
	 */

	//	deviceMotifVariations = *(this->nodeVariations);
	//	this->devicePointerMotifVariations = thrust::raw_pointer_cast(&deviceMotifVariations[0]);
//	int i = 0;
//	//std::cout << "Checker: " << i++ << std::endl;
	gpuErrchk(
			cudaDeviceSetLimit(cudaLimitMallocHeapSize,
					size_t(10) * size_t(numOfNodes) * size_t(numOfNodes)
							* sizeof(int64)));
	size_t currentLimit;
	gpuErrchk(cudaDeviceGetLimit(&currentLimit, cudaLimitMallocHeapSize));

	//std::cout << "Current limit is: " << currentLimit << std::endl;
	gpuErrchk(
			cudaMallocManaged(&(this->devicePointerMotifVariations),
					nodeVariations->size() * sizeof(unsigned int)));
	//	//std::cout << "between"<<std::endl;

	//	//std::cout << this->nodeVariations->data()[0] <<std::endl;

	//	//std::cout << this->nodeVariations->size() <<std::endl;

	std::memcpy(this->devicePointerMotifVariations,
			&((*(this->nodeVariations))[0]),
			nodeVariations->size() * sizeof(unsigned int));
	// Removal index
	//	deviceRemovalIndex = *(this->removalIndex);
	//	//std::cout << "Checker: " << i++ << std::endl;
	gpuErrchk(
			cudaMallocManaged(&(this->devicePointerRemovalIndex),
					removalIndex->size() * sizeof(unsigned int)));
	std::memcpy(this->devicePointerRemovalIndex, this->removalIndex->data(),
			removalIndex->size() * sizeof(unsigned int));
	//Sorted nodes
	//	deviceSortedNodesByDegree = *(this->sortedNodesByDegree);
	//	this->devicePointerSortedNodesByDegree = thrust::raw_pointer_cast(&deviceSortedNodesByDegree[0]);
	//	//std::cout << "Checker: " << i++ << std::endl;
	gpuErrchk(
			cudaMallocManaged(&(this->devicePointerSortedNodesByDegree),
					sortedNodesByDegree->size() * sizeof(unsigned int)));
	std::memcpy(this->devicePointerSortedNodesByDegree,
			this->sortedNodesByDegree->data(),
			sortedNodesByDegree->size() * sizeof(unsigned int));

	// Feature matrix
	//	//std::cout << "Checker: " << i++ << std::endl;
	//std::cout << "Num of Nodes:" << this->numOfNodes << std::endl;
	//std::cout << "Num of node variations: " << this->nodeVariations->size()
			//<< std::endl;
	unsigned int size = this->numOfNodes * this->nodeVariations->size()
			* sizeof(unsigned int);
//	//std::cout << "between" << std::endl;
	gpuErrchk(cudaMallocManaged(&(this->deviceFeatures), size));

	// Original graph
	//	//std::cout << "Checker: " << i++ << std::endl;
	gpuErrchk(
			cudaMallocManaged(&deviceOriginalGraphOffsets,
					(this->numOfNodes + 1) * sizeof(int64)));
	//	//std::cout << "Checker: " << i++ << std::endl;
	gpuErrchk(
			cudaMallocManaged(&deviceOriginalGraphNeighbors,
					(this->numOfEdges) * sizeof(unsigned int)));
	//	//std::cout << "Checker: " << i++ << std::endl;
	std::memcpy(deviceOriginalGraphOffsets, this->mGraph->GetOffsetList(),
			(this->numOfNodes + 1) * sizeof(int64));
	//	//std::cout << "Checker: " << i++ << std::endl;
	std::memcpy(deviceOriginalGraphNeighbors, this->mGraph->GetNeighborList(),
			(this->numOfEdges) * sizeof(unsigned int));

	// Full graph
	//	//std::cout << "Checker: " << i++ << std::endl;
	gpuErrchk(
			cudaMallocManaged(&deviceFullGraphOffsets,
					(this->fullGraph.GetNumberOfNodes() + 1) * sizeof(int64)));
	//	//std::cout << "Checker: " << i++ << std::endl;
	gpuErrchk(
			cudaMallocManaged(&deviceFullGraphNeighbors,
					(this->fullGraph.GetNumberOfEdges())
							* sizeof(unsigned int)));
	//	//std::cout << "Checker: " << i++ << std::endl;
	std::memcpy(deviceFullGraphOffsets, this->fullGraph.GetOffsetList(),
			(this->fullGraph.GetNumberOfNodes() + 1) * sizeof(int64));
	//	//std::cout << "Checker: " << i++ << std::endl;
	std::memcpy(deviceFullGraphNeighbors, this->fullGraph.GetNeighborList(),
			(this->fullGraph.GetNumberOfEdges()) * sizeof(unsigned int));
	//	//std::cout << "Checker: " << i++ << std::endl;

	//Assign to global variables

	globalNumOfNodes = this->numOfNodes;
	globalNumOfMotifs = this->numOfMotifs;
	globalNumOfEdges = this->numOfEdges;
	globalDirected = this->directed;

	globalDevicePointerMotifVariations = this->devicePointerMotifVariations;
	globalDevicePointerRemovalIndex = this->devicePointerRemovalIndex;
	globalDevicePointerSortedNodesByDegree =
			this->devicePointerSortedNodesByDegree;

	globalDeviceOriginalGraphOffsets = this->deviceOriginalGraphOffsets;
	globalDeviceOriginalGraphNeighbors = this->deviceOriginalGraphNeighbors;
	globalDeviceFullGraphOffsets = this->deviceFullGraphOffsets;
	globalDeviceFullGraphNeighbors = this->deviceFullGraphNeighbors;
	globalDeviceFeatures = this->deviceFeatures;

}

// Kernel and friend functions
__global__
void Motif3Kernel(bool* visited) {
	//printf("In motif 3 kernel\n");
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	//printf("In motif 3 kernel\n");
	auto n = globalNumOfNodes;
	//printf("There are %u nodes ",n);
	//printf("in motif 3 kernel\n");
	//AreNeighbors(0,1);
	for (int i = index; i < n; i += stride) {
		//	printf("In motif 3 kernel, i=%i\n",i);
		Motif3Subtree(globalDevicePointerSortedNodesByDegree[i], visited);
	}
}
__global__
void Motif4Kernel(short* visited) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	auto n = globalNumOfNodes;
	for (int i = index; i < n; i += stride)
		Motif4Subtree(globalDevicePointerSortedNodesByDegree[i], visited);
}

vector<vector<unsigned int> *> *GPUMotifCalculator::Calculate() {

	int blockSize = 256;
	int numBlocks = (this->numOfNodes + blockSize - 1) / blockSize;

	//Prefetch all relevant memory
	/*
	 globalDevicePointerMotifVariations = this->devicePointerMotifVariations;
	 globalDevicePointerRemovalIndex = this->devicePointerRemovalIndex;
	 globalDevicePointerSortedNodesByDegree = this -> devicePointerSortedNodesByDegree;

	 globalDeviceOriginalGraphOffsets = this->deviceOriginalGraphOffsets;
	 globalDeviceOriginalGraphNeighbors = this -> deviceOriginalGraphNeighbors;
	 globalDeviceFullGraphOffsets = this->deviceFullGraphOffsets;
	 globalDeviceFullGraphNeighbors = this->deviceFullGraphNeighbors;
	 globalDeviceFeatures = this->deviceFeatures;
	 */
	cudaSetDevice(2);
	int device = -1;
	cudaGetDevice(&device);

	int offsetSize = this->numOfNodes + 1;
	int neighborSize = this->numOfEdges;

	cudaMemPrefetchAsync(globalDevicePointerMotifVariations,
			nodeVariations->size() * sizeof(unsigned int), device, NULL);
	cudaMemPrefetchAsync(globalDevicePointerRemovalIndex,
			this->numOfNodes * sizeof(unsigned int), device, NULL);
	cudaMemPrefetchAsync(globalDevicePointerSortedNodesByDegree,
			this->numOfNodes * sizeof(unsigned int), device, NULL);
	cudaMemPrefetchAsync(globalDeviceOriginalGraphOffsets,
			offsetSize * sizeof(int64), device, NULL);
	cudaMemPrefetchAsync(globalDeviceOriginalGraphNeighbors,
			neighborSize * sizeof(unsigned int), device, NULL);
	cudaMemPrefetchAsync(globalDeviceFullGraphOffsets,
			offsetSize * sizeof(int64), device, NULL);
	cudaMemPrefetchAsync(globalDeviceFullGraphNeighbors,
			neighborSize * sizeof(unsigned int), device, NULL);
	cudaMemPrefetchAsync(globalDeviceFeatures,
			(this->numOfNodes * this->nodeVariations->size())
					* sizeof(unsigned int), device, NULL);

	if (this->level == 3) {
		//std::cout << "Start 3" << std::endl;

		// for (auto node : *(this->sortedNodesByDegree)) {
		// 	////std::cout << node << std::endl;
		// 	Motif3Subtree(node);
		// }
		bool* visited_vertices;
		std::cout << "Num of motifs: "<<globalNumOfMotifs <<std::endl;
		gpuErrchk(cudaMallocManaged(&visited_vertices, numOfNodes * numOfNodes*sizeof(bool)));
		gpuErrchk(cudaMemPrefetchAsync(visited_vertices, numOfNodes * numOfNodes*sizeof(bool),device,NULL));
		Motif3Kernel<<<numBlocks, blockSize>>>(visited_vertices);
		//std::cout << "Starting Motif 3 kernel" << std::endl;
		//Motif3Kernel<<<1,1>>>(this);
	} else {
		////std::cout << "Start 4" << std::endl;

		// for (auto node : *(this->sortedNodesByDegree))
		// 	Motif4Subtree(node);
		short* visited_vertices;
		gpuErrchk(cudaMallocManaged(&visited_vertices, numOfNodes * numOfNodes*sizeof(short)));
		gpuErrchk(cudaMemPrefetchAsync(visited_vertices, numOfNodes * numOfNodes*sizeof(short),device,NULL));
		Motif4Kernel<<<numBlocks, blockSize>>>(visited_vertices);
		//std::cout << "Starting Motif 4 kernel" << std::endl;
		//Motif4Kernel<<<1,1>>>(this);
	}
	////std::cout << "Done All" << std::endl;
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	//TODO: convert the device features to the vector format
	//std::cout << "Num of Motifs: " << this->numOfMotifs << std::endl;
	for (int node = 0; node < this->numOfNodes; node++) {
		for (int motif = 0; motif < this->numOfMotifs; motif++) {
			this->features->at(node)->at(motif) = globalDeviceFeatures[motif
					+ this->numOfMotifs * node];
			//		//std::cout << globalDeviceFeatures[motif + this->numOfMotifs * node]<<"\t";
		}
		//	//std::cout <<std::endl;
	}
	//std::cout << "Num of nodes: " << this->numOfNodes << " features len: "
//			<< this->features->size() << std::endl;
	return this->features;
}

__device__
void Motif3Subtree(unsigned int root, bool* visited) {
	// Instead of yield call GroupUpdater function
	// Don't forget to check each time that the nodes are in the graph (check removal index).
	int checker = 0;
	//printf("Motif 3 checker: %i\n",checker++);
	int idx_root = globalDevicePointerRemovalIndex[root];// root_idx is also our current iteration -
//	bool* visited_vertices = (bool*) malloc(globalNumOfNodes); // every node_idx smaller than root_idx is already handled

	// For test graphs that are regular with d=20, 500 is enough
	bool* visited_vertices = visited + idx_root * globalNumOfNodes; // every node_idx smaller than root_idx is already handled
	if (visited_vertices == NULL)
		printf(
				"Error: No more memory to allocate visited vertices in node %u\n",
				root);
	//printf("idx_root=%d\n",idx_root);
	for (int i = 0; i < globalNumOfNodes; i++)
		visited_vertices[i] = false;
	visited_vertices[root] = true;

	//	printf("%i\t",checker++);
	const unsigned int *neighbors = globalDeviceFullGraphNeighbors; // all neighbors - ancestors and descendants
	const int64 *offsets = globalDeviceFullGraphOffsets;

	// TODO problem with dual edges
	////std::cout << "Mark" << std::endl;
	for (int64 i = offsets[root]; i < offsets[root + 1]; i++) // loop first neighbors
		if (globalDevicePointerRemovalIndex[neighbors[i]] > idx_root) // n1 not handled yet
			visited_vertices[neighbors[i]] = true;
	////std::cout << "Mark" << std::endl;
	//printf("Motif 3 checker: %i\n",checker++);
	//printf("Offsets: %u  - %u \n",offsets[root],offsets[root+1]);
	for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
		unsigned int n1 = neighbors[n1_idx];
		//	printf("n1=%u\n",n1);
		if (globalDevicePointerRemovalIndex[n1] <= idx_root) // n1 already handled
			continue;
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors
			unsigned int n2 = neighbors[n2_idx];
			//		printf("n2=%u\n",n2);
			if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
				continue;
			if (visited_vertices[n2]) {			// check if n2 was visited &&
				//			printf("n2 was seen");
				bool edgeExists = AreNeighbors(root,n2) || AreNeighbors(n2,root);
				if (!edgeExists || (edgeExists && n1 < n2)) {// n2 is after n1 (stops counting the motif twice)
					unsigned int arr[] = { root, n1, n2 };
					GroupUpdater(arr, 3); // update motif counter [r,n1,n2]
				}
			} else {
				visited_vertices[n2] = true;
				unsigned int arr[] = { root, n1, n2 };
				GroupUpdater(arr, 3); // update motif counter [r,n1,n2]
			}										   // end ELSE
		}											// end LOOP_SECOND_NEIGHBORS

	} // end LOOP_FIRST_NEIGHBORS
	 // std::cout << "Mark" << std::endl;
	  // vector<vector<unsigned int> *> *n1_comb = neighbors_combinations(neighbors,
	  // 	offsets[root], offsets[root + 1]);
	  //printf("Motif 3 checker: %i\n",checker++);
	for (int64 i = offsets[root]; i < offsets[root + 1]; i++) {
		for (int64 j = i + 1; j < offsets[root + 1]; j++) {
			unsigned int n1 = neighbors[i];
			unsigned int n2 = neighbors[j];
			////std::cout << "\t" << n1 << "," << n2 << std::endl;
			if (globalDevicePointerRemovalIndex[n1] <= idx_root
					|| globalDevicePointerRemovalIndex[n2] <= idx_root) // motif already handled
				continue;
			////std::cout << "Mark1" << std::endl;
			////std::cout << (visited_vertices[n1] < visited_vertices[n2]) << std::endl;
			////std::cout << mGraph->areNeighbors(n1, n2) << std::endl;
			////std::cout << mGraph->areNeighbors(n2, n1) << std::endl;
			if ((n1 < n2) && !(AreNeighbors(n1, n2) || AreNeighbors(n2, n1))) { // check n1, n2 not neighbors
				////std::cout << "Mark2" << std::endl;
				unsigned int arr[] = { root, n1, n2 };
				GroupUpdater(arr, 3); // update motif counter [r,n1,n2]
			}
		}
	} // end loop COMBINATIONS_NEIGHBORS_N1
	  //free(visited_vertices);
}

__device__
void Motif4Subtree(unsigned int root, short* visited) {
	int idx_root = globalDevicePointerRemovalIndex[root]; // root_idx is also our current iteration -
	short* visited_vertices = visited + idx_root * globalNumOfNodes; // every node_idx smaller than root_idx is already handled
	for (int i = 0; i < globalNumOfNodes; i++)
		visited_vertices[i] = -1;
	visited_vertices[root] = 0;

	const unsigned int *neighbors = globalDeviceFullGraphNeighbors; // all neighbors - ancestors and descendants
	const int64 *offsets = globalDeviceFullGraphOffsets;

	// TODO problem with dual edges
	for (int64 i = offsets[root]; i < offsets[root + 1]; i++) // loop first neighbors
		if (globalDevicePointerRemovalIndex[neighbors[i]] > idx_root) // n1 not handled yet
			visited_vertices[neighbors[i]] = 1;

	/*
	 *    for n1, n2, n3 in combinations(neighbors_first_deg, 3):
	 yield [root, n1, n2, n3]
	 */
	// vector<vector<unsigned int> *> *n1_3_comb = neighbors_combinations(
	// 	neighbors, offsets[root], offsets[root + 1], 3);
	// for (auto it = n1_3_comb->begin(); it != n1_3_comb->end(); ++it)
	// {
	int64 end = offsets[root + 1];
	for (int64 i = offsets[root]; i < end; i++) {
		for (int64 j = i + 1; j < end; j++) {
			if (j == end - 1) //if j is the last element, we can't add an element and therefore it's not a 3-combination
				continue;
			for (int64 k = j + 1; k < end; k++) {
				unsigned int n11 = neighbors[i];
				unsigned int n12 = neighbors[j];
				unsigned int n13 = neighbors[k];
				if (globalDevicePointerRemovalIndex[n11] <= idx_root
						|| globalDevicePointerRemovalIndex[n12] <= idx_root
						|| globalDevicePointerRemovalIndex[n13] <= idx_root) // motif already handled
					continue;
				unsigned int arr[] = { root, n11, n12, n13 };
				GroupUpdater(arr, 4); // update motif counter [r,n11,n12,n13]
			}
		}
	}

	// All other cases
	for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
		unsigned int n1 = neighbors[n1_idx];
		if (globalDevicePointerRemovalIndex[n1] <= idx_root) // n1 already handled
			continue;
		//Mark second neighbors
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors
			unsigned int n2 = neighbors[n2_idx];
			if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
				continue;
			if (visited_vertices[n2] == -1) { // check if n2 was *not* visited
				visited_vertices[n2] = 2;

			} //end if
		}	 //end loop SECOND NEIGHBORS

		// The case of root-n1-n2-n11
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (again)
			unsigned int n2 = neighbors[n2_idx];
			if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
				continue;
			for (int64 n11_idx = offsets[root]; n11_idx < offsets[root + 1];
					n11_idx++) { // loop first neighbors
				unsigned int n11 = neighbors[n11_idx];
				if (globalDevicePointerRemovalIndex[n11] <= idx_root) // n2 already handled
					continue;
				if (visited_vertices[n2] == 2 && n11 != n1) {
					bool edgeExists = AreNeighbors(n2, n11)
							|| AreNeighbors(n11, n2);
					if (!edgeExists || (edgeExists && n1 < n11)) {
						unsigned int arr[] = { root, n1, n11, n2 };
						GroupUpdater(arr, 4); // update motif counter [r,n1,n11,n2]
					}												// end if
				}
			}								// end loop INNER FIRST NEIGHBORS
		}									// end loop SECOND NEIGHBORS AGAIN

		// The case of root-n1-n21-n22
		//2-combinations on second neighbors
		end = offsets[n1 + 1];
		for (int64 i = offsets[n1]; i < end; i++) {
			for (int64 j = i + 1; j < end; j++) {

				unsigned int n21 = neighbors[i];
				unsigned int n22 = neighbors[j];
				if (globalDevicePointerRemovalIndex[n21] <= idx_root
						|| globalDevicePointerRemovalIndex[n22] <= idx_root) // motif already handled
					continue;
				if (2 == visited_vertices[n21] && visited_vertices[n22] == 2) {
					unsigned int arr[] = { root, n1, n21, n22 };
					GroupUpdater(arr, 4); // update motif counter [r,n1,n21,n22]
				}
			}
		} // end loop SECOND NEIGHBOR COMBINATIONS
	}
	//The case of n1-n2-n3
	for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
		unsigned int n1 = neighbors[n1_idx];
		if (globalDevicePointerRemovalIndex[n1] <= idx_root) // n1 already handled
			continue;
		for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (third time's the charm)
			unsigned int n2 = neighbors[n2_idx];
			if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
				continue;

			if (visited_vertices[n2] == 1)
				continue;

			for (int64 n3_idx = offsets[n2]; n3_idx < offsets[n2 + 1];
					n3_idx++) { // loop third neighbors
				unsigned int n3 = neighbors[n3_idx];
				if (globalDevicePointerRemovalIndex[n3] <= idx_root) // n2 already handled
					continue;
				if (visited_vertices[n3] == 1)
					continue;

				if (visited_vertices[n3] == -1) { // check if n3 was *not* visited
					visited_vertices[n3] = 3;
					if (visited_vertices[n2] == 2) { // check if n2 is a visited second neighbor
						unsigned int arr[] = { root, n1, n2, n3 };
						GroupUpdater(arr, 4); // update motif counter [r,n1,n2,n3]
					}		// end check if n2 is a visited second neighbor
				}						// end check if n3 was not visited

				else {

					if (visited_vertices[n3] == 2
							&& !(AreNeighbors(n1, n3) || AreNeighbors(n3, n1))) {
						unsigned int arr[] = { root, n1, n2, n3 };
						GroupUpdater(arr, 4); // update motif counter [r,n1,n2,n3]
					}
					if (visited_vertices[n3] == 3
							&& visited_vertices[n2] == 2) {
						unsigned int arr[] = { root, n1, n2, n3 };
						GroupUpdater(arr, 4); // update motif counter [r,n1,n2,n3]
					}											   // end if
				}											   //end else
			}									// end loop THIRD NEIGHBORS
		}				// end loop SECOND NEIGHBORS THIRD TIME'S THE CHARM

	} // end loop FIRST NEIGHBORS
	  //free(visited_vertices);

}

__device__
bool AreNeighbors(unsigned int p, unsigned int q) {
	// int64* deviceOriginalGraphOffsets;
	// unsigned int* deviceOriginalGraphNeighbors;
	 int first = globalDeviceOriginalGraphOffsets[p],//first array element
			last = globalDeviceOriginalGraphOffsets[p + 1] - 1,	//last array element
			middle;		//mid point of search

	while (first <= last) {
		middle = (int)(first + last) / 2; //this finds the mid point
		////std::cout << "Binary search: " << middle << std::endl;
		//TODO: fix overflow problem
		if (globalDeviceOriginalGraphNeighbors[middle] == q) {
			return true;
		} else if (globalDeviceOriginalGraphNeighbors[middle] < q)
				{
			first = middle + 1;      //if it's in the upper half

		} else {
			last = middle - 1; // if it's in the lower half
		}
	}
	return false;  // not found

}

__device__
void GroupUpdater(unsigned int group[], int size) {
	// TODO: count overall number of motifs in graph (maybe different class)?
	//printf("In GroupUpdater");
	
	int groupNumber = GetGroupNumber(group, size);
	int motifNumber = (globalDevicePointerMotifVariations)[groupNumber];
//	if(size==3){
		//std::cout << motifNumber <<","<<group[0]<<","<<group[1]<<","<<group[2]<<std::endl;
//		printf("%d,%u,%u,%u\n",motifNumber,group[0],group[1],group[2]);
//	}

	if (motifNumber != -1) {
		//	printf("Found motif!\n");
		for (int i = 0; i < size; i++)
			atomicAdd(
					globalDeviceFeatures
							+ (motifNumber + globalNumOfMotifs * group[i]), 1);	//atomic add + access as 1D array : features[motif + M*node] // @suppress("Function cannot be resolved")
		// where M is the number of motifs
	}
}

__device__
int GetGroupNumber(unsigned int group[], int size) {
	int sum = 0;
	int power = 1;
	bool hasEdge;
	if (globalDirected) {
		// Use permutations
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (i != j) {
					hasEdge = AreNeighbors(group[i], group[j]);
					if (hasEdge)
						sum += power;
					power *= 2;
				}
			}
		}
	} else {
		// Use combinations
		for (int i = 0; i < size; i++) {
			for (int j = i + 1; j < size; j++) {

				hasEdge = AreNeighbors(group[i], group[j]);
				if (hasEdge)
					sum += power;
				power *= 2;
			}
		}

	}
	return sum;
}

GPUMotifCalculator::~GPUMotifCalculator() {
	//map the group num to the iso motif
	delete nodeVariations;
	//the index in which we remove the node from the graph. Basically, from this index on the node doesen't exist.
	delete removalIndex;
	//the nodes, sorted in descending order by the degree.
	delete sortedNodesByDegree;

	// Memory resources

	cudaFree(devicePointerMotifVariations);
	cudaFree(devicePointerRemovalIndex);
	cudaFree(devicePointerSortedNodesByDegree);
	// For the original graph
	cudaFree(deviceOriginalGraphOffsets); // @suppress("Function cannot be resolved")
	cudaFree(deviceOriginalGraphNeighbors); // @suppress("Function cannot be resolved")

	// For the full graph
	cudaFree(deviceFullGraphOffsets); // @suppress("Function cannot be resolved")
	cudaFree(deviceFullGraphNeighbors); // @suppress("Function cannot be resolved")

	// Feature array
	cudaFree(deviceFeatures); // @suppress("Function cannot be resolved")
}
