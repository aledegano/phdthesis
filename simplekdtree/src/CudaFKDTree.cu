#include <cuda.h>
#include <host_defines.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <CudaKDtree/KDTree/src/CudaFKDTree.cuh>
#include "CUDAQueue.h"
#include "eclipse_cuda_parser.h"

__host__ __device__
bool intersects(unsigned int index, CudaFKDPoint minPoint, CudaFKDPoint maxPoint, int dim, const CudaFKDPoint* theDimensions) {
	return (theDimensions[index][dim] <= maxPoint[dim] && theDimensions[index][dim] >= minPoint[dim]);
}

__host__ __device__
bool isInTheBox(unsigned int index, CudaFKDPoint minPoint, CudaFKDPoint maxPoint, const CudaFKDPoint* theDimensions) {
	bool inTheBox = true;
	for (int dim = 0; dim < 3; ++dim)
		inTheBox &= (theDimensions[index][dim] <= maxPoint[dim] && theDimensions[index][dim] >= minPoint[dim]);
	return inTheBox;
}

__global__
void kernel_search_in_the_box_linear(unsigned int offset, const CudaFKDPoint* searchBox, unsigned int maxNN, CudaFKDPoint* theDimensions, const unsigned int theNumberOfPoints, unsigned int* result) {
	unsigned int thd_id = offset + threadIdx.x + blockDim.x * blockIdx.x; // Offset accounts for the stream the kernel is in
	if (thd_id > theNumberOfPoints)
		return;
	CudaFKDPoint minPoint = searchBox[2 * thd_id];
	CudaFKDPoint maxPoint = searchBox[(2 * thd_id) + 1];
	constexpr const unsigned int maxAddresses = 42;
	CUDAQueue<maxAddresses, unsigned int> indecesToVisit;
	indecesToVisit.push_singleThread(0);
	unsigned int index = 0;
	unsigned int dimension = 0;
	unsigned int depth = 0;
	unsigned int resultSize = 1;
	int pushRes = 0;
	while (indecesToVisit.m_size > 0) {
		index = indecesToVisit.pop_back_singleThread();
		depth = ((unsigned int) (31 - __clz((index + 1) | 1)));
		dimension = depth % numberOfDimensions;
		bool intersection = intersects(index, minPoint, maxPoint, dimension, theDimensions);
		if (intersection && isInTheBox(index, minPoint, maxPoint, theDimensions)) {
			if (resultSize > (maxNN - 1))
				break;
			result[thd_id * maxNN + resultSize] = theDimensions[index].getId(); //The id of the node is the index of the point wrt the original ordering
			resultSize++;
		}
		bool isLowerThanBoxMin = theDimensions[index][dimension] < minPoint[dimension];
		int startSon = isLowerThanBoxMin; //left son = 0, right son =1
		int endSon = isLowerThanBoxMin || intersection;
		for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
			unsigned int indexToAdd = 2 * index + 1 + whichSon;
			if (indexToAdd < theNumberOfPoints)
				pushRes = indecesToVisit.push_singleThread(indexToAdd);
			if (pushRes < 0) {
				printf("Error CUDAQueue exceed size.");
				return;
			}
		}
	}
	result[thd_id * maxNN] = resultSize - 1; // Use the first element to tell how many results has been found
}

void CudaFKDTree::search_in_the_box_linear(CudaFKDPoint* searchBox, unsigned int maxNN, unsigned int* host_results, const unsigned int nStreams, const unsigned int threads_per_block) const {
	CudaFKDPoint* device_searchBox;
	unsigned int* device_results;
	CudaSafeCall(cudaMalloc((void ** ) &device_searchBox, theNumberOfPoints * 2 * sizeof(CudaFKDPoint)));
	CudaSafeCall(cudaMalloc((void ** ) &device_results, theNumberOfPoints * maxNN * sizeof(unsigned int))); // This will contain the IDs of the NODES, need to be converted to point ID before returning
	cudaStream_t stream[nStreams];
	for (unsigned int i = 0; i < nStreams; i++)
		CudaSafeCall(cudaStreamCreate(&stream[i]));
	unsigned int streamSize = theNumberOfPoints / nStreams;
	for (unsigned int i = 0; i < nStreams; i++) {
		unsigned int offset = i * streamSize;
		if (i == nStreams - 1)
			streamSize = streamSize + (theNumberOfPoints % nStreams); // The last stream gets the remainder of the points
		CudaSafeCall(cudaMemcpyAsync(&device_searchBox[2 * offset], &searchBox[2 * offset], streamSize * 2 * sizeof(CudaFKDPoint), cudaMemcpyHostToDevice, stream[i]));
		kernel_search_in_the_box_linear<<<(streamSize + threads_per_block - 1) / threads_per_block, threads_per_block, 0, stream[i]>>>(offset, device_searchBox, maxNN, device_theDimensions, theNumberOfPoints, device_results);
		CudaSafeCall(cudaMemcpyAsync(&host_results[offset * maxNN], &device_results[offset * maxNN], streamSize * maxNN * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[i]));
	}
	for (unsigned k = 0; k < nStreams; k++)
		cudaStreamSynchronize(stream[k]);
	CudaSafeCall(cudaFree(device_searchBox));
	CudaSafeCall(cudaFree(device_results));
	return;
}

#define FLOOR_LOG2(X) ((unsigned int) (31 - __builtin_clz( (X) | 1) ))
std::vector<std::vector<unsigned int> > CudaFKDTree::search_in_the_box_sequential(CudaFKDPoint* searchBox) const {
	std::vector<std::vector<unsigned int> > result(theNumberOfPoints);
	for (unsigned int id = 0; id < theNumberOfPoints; id++) {
		CudaFKDPoint minPoint = searchBox[2 * id];
		CudaFKDPoint maxPoint = searchBox[(2 * id) + 1];
		std::vector<unsigned int> indecesToVisit;
		result.reserve(16);
		indecesToVisit.push_back(0);
		unsigned int index = 0;
		unsigned int dimension = 0;
		unsigned int depth = 0;
		while (!indecesToVisit.empty()) {
			index = indecesToVisit.back();
			indecesToVisit.pop_back();
			depth = FLOOR_LOG2(index + 1);
			dimension = depth % numberOfDimensions;
			bool intersection = intersects(index, minPoint, maxPoint, dimension, theDimensions.data());
			if (intersection && isInTheBox(index, minPoint, maxPoint, theDimensions.data())) {
				result[id].push_back(theDimensions[index].getId());
			}
			bool isLowerThanBoxMin = theDimensions[dimension][index] < minPoint[dimension];
			int startSon = isLowerThanBoxMin; //left son = 0, right son =1
			int endSon = isLowerThanBoxMin || intersection;
			for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
				unsigned int indexToAdd = leftSonIndex(index) + whichSon;
				if (indexToAdd < theNumberOfPoints)
					indecesToVisit.push_back(indexToAdd);
			}
		}
	}
	return result;
}

unsigned int CudaFKDTree::leftSonIndex(unsigned int index) const {
	return 2 * index + 1;
}

unsigned int CudaFKDTree::rightSonIndex(unsigned int index) const {
	return 2 * index + 2;
}
