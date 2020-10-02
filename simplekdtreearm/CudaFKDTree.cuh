#ifndef CUDAFKDTREE_FKDTREE_H_
#define CUDAFKDTREE_FKDTREE_H_

#include "CudaFKDPoint.h"
#include "cudaError.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <utility>
#include <iostream>
#include <deque>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

#include <host_defines.h>

class CudaFKDTree {
public:
	CudaFKDTree(const long int nPoints) :
			theNumberOfPoints(nPoints) {
		theDepth = std::floor(log2((double) nPoints));
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints);
		thePoints.reserve(theNumberOfPoints);
		device_theDimensions = NULL;
	}
	CudaFKDTree(const long int nPoints, const std::vector<CudaFKDPoint>& points) :
			theNumberOfPoints(nPoints) {
		theDepth = std::floor(log2((double) nPoints));
		theDimensions.resize(nPoints);
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints, 0);
		thePoints = points;
		device_theDimensions = NULL;
	}

	CudaFKDTree(unsigned int capacity);
	CudaFKDTree(const CudaFKDTree& v);

	~CudaFKDTree() {
		if (device_theDimensions != NULL)
			CudaSafeCall(cudaFree(device_theDimensions));
	}
	void build() {
		//gather kdtree building
		int dimension;
		theIntervalMin[0] = 0;
		theIntervalLength[0] = theNumberOfPoints;
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();
		for (int depth = 0; depth < theDepth; ++depth) {
			dimension = depth % numberOfDimensions;
			unsigned int firstIndexInDepth = (1 << depth) - 1;
			for (int indexInDepth = 0; indexInDepth < (1 << depth); ++indexInDepth) {
				unsigned int indexInArray = firstIndexInDepth + indexInDepth;
				unsigned int leftSonIndexInArray = 2 * indexInArray + 1;
				unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
				unsigned int whichElementInInterval = partition_complete_kdtree(theIntervalLength[indexInArray]);
				std::nth_element(thePoints.begin() + theIntervalMin[indexInArray], thePoints.begin() + theIntervalMin[indexInArray] + whichElementInInterval,
						thePoints.begin() + theIntervalMin[indexInArray] + theIntervalLength[indexInArray], [dimension](const CudaFKDPoint& a, const CudaFKDPoint& b) -> bool {
							if(a[dimension] == b[dimension])
							return a.getId() < b.getId();
							else
							return a[dimension] < b[dimension];
						});
				add_at_position(thePoints[theIntervalMin[indexInArray] + whichElementInInterval], indexInArray);
				if (leftSonIndexInArray < theNumberOfPoints) {
					theIntervalMin[leftSonIndexInArray] = theIntervalMin[indexInArray];
					theIntervalLength[leftSonIndexInArray] = whichElementInInterval;
				}
				if (rightSonIndexInArray < theNumberOfPoints) {
					theIntervalMin[rightSonIndexInArray] = theIntervalMin[indexInArray] + whichElementInInterval + 1;
					theIntervalLength[rightSonIndexInArray] = (theIntervalLength[indexInArray] - 1 - whichElementInInterval);
				}
			}
		}
		end = std::chrono::system_clock::now();
		float node_find_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[CudaTree] Node set in: " << node_find_time << " ms." << std::endl;
		dimension = theDepth % numberOfDimensions;
		unsigned int firstIndexInDepth = (1 << theDepth) - 1;
		for (unsigned int indexInArray = firstIndexInDepth; indexInArray < theNumberOfPoints; ++indexInArray)
			add_at_position(thePoints[theIntervalMin[indexInArray]], indexInArray);
		// Build complete. Move the nodes on the GPU global memory
		cudaEvent_t cudaStart, cudaStop;
		float node_move_time = 0;
		cudaEventCreate(&cudaStart);
		cudaEventCreate(&cudaStop);
		cudaEventRecord(cudaStart);
		CudaSafeCall(cudaMalloc((void ** ) &device_theDimensions, theNumberOfPoints * sizeof(CudaFKDPoint)));
		CudaSafeCall(cudaMemcpy(device_theDimensions, theDimensions.data(), theNumberOfPoints * sizeof(CudaFKDPoint), cudaMemcpyHostToDevice));
		cudaEventRecord(cudaStop);
		cudaEventSynchronize(cudaStop);
		cudaEventElapsedTime(&node_move_time, cudaStart, cudaStop);
		std::cout << "[CudaFKDTree] Node copy on GPU time: " << node_move_time << " ms." << std::endl;
	}

	std::vector<unsigned int> search_in_the_box(const CudaFKDPoint&, const CudaFKDPoint&) const;
	void search_in_the_box_linear(CudaFKDPoint*, unsigned int, unsigned int*, const unsigned int nStreams = 8, const unsigned int threads_per_block = 32, const unsigned int burnoutRepetitions = 1) const;

	void add_at_position(const CudaFKDPoint& point, const unsigned int position) {
		for (unsigned int dim = 0; dim < numberOfDimensions; ++dim)
			theDimensions[position][dim] = point[dim];
		theDimensions[position].setId(point.getId());
		theIds[position] = point.getId();

	}
	void add_at_position(CudaFKDPoint && point, const unsigned int position) {
		for (unsigned int dim = 0; dim < numberOfDimensions; ++dim)
			theDimensions[position][dim] = point[dim];
		theDimensions[position].setId(point.getId());
		theIds[position] = point.getId();

	}
	CudaFKDPoint getPoint(unsigned int index) const {
		CudaFKDPoint point;
		for (unsigned int dim = 0; dim < numberOfDimensions; ++dim)
			point.setDimension(dim, theDimensions[index][dim]);
		point.setId(theIds[index]);
		return point;
	}
	bool test_correct_build(unsigned int index = 0, int dimension = 0) const {
		bool correct = true;
		unsigned int leftSonIndexInArray = 2 * index + 1;
		unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
		if (rightSonIndexInArray >= theNumberOfPoints && leftSonIndexInArray >= theNumberOfPoints) {
			return true;
		} else {
			if (leftSonIndexInArray < theNumberOfPoints) {
				if (theDimensions[index][dimension] >= theDimensions[leftSonIndexInArray][dimension]) {
					correct &= test_correct_build(leftSonIndexInArray, (dimension + 1) % numberOfDimensions);
				} else
					return false;
			}
			if (rightSonIndexInArray < theNumberOfPoints) {
				if (theDimensions[index][dimension] <= theDimensions[rightSonIndexInArray][dimension]) {
					correct &= test_correct_build(rightSonIndexInArray, (dimension + 1) % numberOfDimensions);
				} else
					return false;
			}
		}
		return correct;
	}
	bool test_correct_search(const std::vector<unsigned int> foundPoints, const CudaFKDPoint& minPoint, const CudaFKDPoint& maxPoint) const {
		bool testGood = true;
		for (unsigned int i = 0; i < theNumberOfPoints; ++i) {
			bool shouldBeInTheBox = true;
			for (unsigned int dim = 0; dim < numberOfDimensions; ++dim) {
				shouldBeInTheBox &= (thePoints[i][dim] <= maxPoint[dim] && thePoints[i][dim] >= minPoint[dim]);
			}
			bool foundToBeInTheBox = std::find(foundPoints.begin(), foundPoints.end(), thePoints[i].getId()) != foundPoints.end();
			if (foundToBeInTheBox == shouldBeInTheBox) {
				testGood &= true;
			} else {
				if (foundToBeInTheBox)
					std::cerr << "Point " << thePoints[i].getId() << " was wrongly found to be in the box." << std::endl;
				else
					std::cerr << "Point " << thePoints[i].getId() << " was wrongly found to be outside the box." << std::endl;
				testGood &= false;
			}
		}
		if (testGood)
			std::cout << "Search correctness test completed successfully." << std::endl;
		return testGood;
	}

	std::vector<unsigned int> getIdVector() const {
		return theIds;
	}

private:
	unsigned int partition_complete_kdtree(unsigned int length) {
		if (length == 1)
			return 0;
		unsigned int index = 1 << ((int) log2((double) length));
		if ((index / 2) - 1 <= length - index)
			return index - 1;
		else
			return length - index / 2;
	}
	unsigned int leftSonIndex(unsigned int index) const;
	unsigned int rightSonIndex(unsigned int index) const;
	bool intersects(unsigned int index, const CudaFKDPoint& minPoint, const CudaFKDPoint& maxPoint, int dimension) const;
	bool isInTheBox(unsigned int index, const CudaFKDPoint& minPoint, const CudaFKDPoint& maxPoint) const;
	const unsigned int theNumberOfPoints;
	int theDepth;
	std::vector<CudaFKDPoint> thePoints;
	std::vector<CudaFKDPoint> theDimensions;
	CudaFKDPoint* device_theDimensions; // Cuda copy of the tree nodes
	std::vector<unsigned int> theIntervalLength;
	std::vector<unsigned int> theIntervalMin;
	std::vector<unsigned int> theIds;
};

#endif /* FKDTREE_FKDTREE_H_ */
