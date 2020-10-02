#ifndef FKDTREE_FKDTREE_H_
#define FKDTREE_FKDTREE_H_

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <utility>
#include <iostream>
#include <deque>
#include <cassert>
#include <cmath>

#include <chrono>

#include "FKDPoint.h"
#include "FQueue.h"

template<class TYPE, int numberOfDimensions>
class FKDTree {

public:

	FKDTree(const long int nPoints) {
		theNumberOfPoints = nPoints;
		theDepth = std::floor(log2(nPoints));
		for (auto& x : theDimensions)
			x.resize(theNumberOfPoints);
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints);
		thePoints.reserve(theNumberOfPoints);

	}

	FKDTree(const long int nPoints, const std::vector<FKDPoint<TYPE, numberOfDimensions> >& points) {
		theNumberOfPoints = nPoints;
		theDepth = std::floor(log2(nPoints));
		for (auto& x : theDimensions)
			x.resize(theNumberOfPoints);
		theIntervalLength.resize(theNumberOfPoints, 0);
		theIntervalMin.resize(theNumberOfPoints, 0);
		theIds.resize(theNumberOfPoints, 0);
		thePoints = points;
	}

	FKDTree() {
		theNumberOfPoints = 0;
		theDepth = 0;
		for (auto& x : theDimensions)
			x.clear();
		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
	}

	FKDTree(unsigned int capacity);

	FKDTree(const FKDTree<TYPE, numberOfDimensions>& v);

	FKDTree(FKDTree<TYPE, numberOfDimensions> && other) {
		theNumberOfPoints(std::move(other.theNumberOfPoints));
		theDepth(std::move(other.theDepth));

		theIntervalLength.clear();
		theIntervalMin.clear();
		theIds.clear();
		thePoints.clear();
		for (auto& x : theDimensions)
			x.clear();

		theIntervalLength = std::move(other.theIntervalLength);
		theIntervalMin = std::move(other.theIntervalMin);
		theIds = std::move(other.theIds);

		thePoints = std::move(other.thePoints);
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions = std::move(other.theDimensions);
	}

	FKDTree<TYPE, numberOfDimensions>& operator=(FKDTree<TYPE, numberOfDimensions> && other) {

		if (this != &other) {
			theNumberOfPoints(std::move(other.theNumberOfPoints));
			theDepth(std::move(other.theDepth));

			theIntervalLength.clear();
			theIntervalMin.clear();
			theIds.clear();
			thePoints.clear();
			for (auto& x : theDimensions)
				x.clear();

			theIntervalLength = std::move(other.theIntervalLength);
			theIntervalMin = std::move(other.theIntervalMin);
			theIds = std::move(other.theIds);

			thePoints = std::move(other.thePoints);
			for (int i = 0; i < numberOfDimensions; ++i)
				theDimensions = std::move(other.theDimensions);
		}
		return *this;

	}

	void push_back(const FKDPoint<TYPE, numberOfDimensions>& point) {

		thePoints.push_back(point);
#pragma unroll
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions.at(i).push_back(point[i]);
		theIds.push_back(point.getId());
	}

	void push_back(FKDPoint<TYPE, numberOfDimensions> && point) {

		thePoints.push_back(point);
#pragma unroll
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions.at(i).push_back(point[i]);
		theIds.push_back(point.getId());
	}

	void add_at_position(const FKDPoint<TYPE, numberOfDimensions>& point, const unsigned int position) {
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions[i][position] = point[i];
		theIds[position] = point.getId();

	}

	void add_at_position(FKDPoint<TYPE, numberOfDimensions> && point, const unsigned int position) {
		for (int i = 0; i < numberOfDimensions; ++i)
			theDimensions[i][position] = point[i];
		theIds[position] = point.getId();

	}

	FKDPoint<TYPE, numberOfDimensions> getPoint(unsigned int index) const {
		FKDPoint<TYPE, numberOfDimensions> point;
		for (int i = 0; i < numberOfDimensions; ++i)
			point.setDimension(i, theDimensions[i][index]);
		point.setId(theIds[index]);
		return point;
	}

	std::vector<unsigned int> search_in_the_box(const FKDPoint<TYPE, numberOfDimensions>&, const FKDPoint<TYPE, numberOfDimensions>&) const;
	std::vector<unsigned int> search_in_the_box_linear(const FKDPoint<TYPE, numberOfDimensions>&, const FKDPoint<TYPE, numberOfDimensions>&) const;

	bool test_correct_build(unsigned int index = 0, int dimension = 0) const {
		bool correct = true;
		unsigned int leftSonIndexInArray = 2 * index + 1;
		unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
		if (leftSonIndexInArray >= theNumberOfPoints) {
			return true;
		} else {
			if (leftSonIndexInArray < theNumberOfPoints) {
				if (theDimensions[dimension][index] >= theDimensions[dimension][leftSonIndexInArray]) {
					correct &= test_correct_build(leftSonIndexInArray, (dimension + 1) % numberOfDimensions);
				} else
					return false;
			}
			if (rightSonIndexInArray < theNumberOfPoints) {
				if (theDimensions[dimension][index] <= theDimensions[dimension][rightSonIndexInArray]) {
					correct &= test_correct_build(rightSonIndexInArray, (dimension + 1) % numberOfDimensions);

				} else
					return false;
			}
		}
		return correct;
	}

	bool test_correct_search(const std::vector<unsigned int> foundPoints, const FKDPoint<TYPE, numberOfDimensions>& minPoint, const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
		bool testGood = true;
		for (unsigned int i = 0; i < theNumberOfPoints; ++i) {
			bool shouldBeInTheBox = true;
			for (int dim = 0; dim < numberOfDimensions; ++dim) {
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

	std::vector<TYPE> getDimensionVector(const int dimension) const {
		assert(dimension < numberOfDimensions);
		return theDimensions[dimension];
	}

	std::vector<unsigned int> getIdVector() const {
		return theIds;
	}
	void build();
private:

	unsigned int partition_complete_kdtree(unsigned int length) {
		if (length == 1)
			return 0;
		unsigned int index = 1 << ((int) log2(length));
		if ((index / 2) - 1 <= length - index)
			return index - 1;
		else
			return length - index / 2;
	}
	unsigned int leftSonIndex(unsigned int index) const;
	unsigned int rightSonIndex(unsigned int index) const;
	bool intersects(unsigned int index, const FKDPoint<TYPE, numberOfDimensions>& minPoint, const FKDPoint<TYPE, numberOfDimensions>& maxPoint, int dimension) const;
	bool isInTheBox(unsigned int index, const FKDPoint<TYPE, numberOfDimensions>& minPoint, const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const;
	unsigned int theNumberOfPoints;
	int theDepth;
	std::vector<FKDPoint<TYPE, numberOfDimensions> > thePoints;
	std::array<std::vector<TYPE>, numberOfDimensions> theDimensions;
	std::vector<unsigned int> theIntervalLength;
	std::vector<unsigned int> theIntervalMin;
	std::vector<unsigned int> theIds;

};

template<class TYPE, int numberOfDimensions>
std::vector<unsigned int> FKDTree<TYPE, numberOfDimensions>::search_in_the_box(const FKDPoint<TYPE, numberOfDimensions>& minPoint, const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
	FQueue<unsigned int> indecesToVisit(128);
	std::vector<unsigned int> result;
	result.reserve(16);
	indecesToVisit.push_back(0);
	for (int depth = 0; depth < theDepth + 1; ++depth) {
		int dimension = depth % numberOfDimensions;
		unsigned int numberOfIndecesToVisitThisDepth = indecesToVisit.size();
		for (unsigned int visitedIndecesThisDepth = 0; visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth; visitedIndecesThisDepth++) {
			unsigned int index = indecesToVisit[visitedIndecesThisDepth];
			bool intersection = intersects(index, minPoint, maxPoint, dimension);
			if (intersection && isInTheBox(index, minPoint, maxPoint))
				result.push_back(theIds[index]);
			bool isLowerThanBoxMin = theDimensions[dimension][index] < minPoint[dimension];
			int startSon = isLowerThanBoxMin; //left son = 0, right son =1
			int endSon = isLowerThanBoxMin || intersection;
			for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
				unsigned int indexToAdd = leftSonIndex(index) + whichSon;
				if (indexToAdd < theNumberOfPoints)
					indecesToVisit.push_back(indexToAdd);
			}
		}
		indecesToVisit.pop_front(numberOfIndecesToVisitThisDepth);
	}
	return result;
}
#define FLOOR_LOG2(X) ((unsigned int) (31 - __builtin_clz( (X) | 1) ))

template<class TYPE, int numberOfDimensions>
std::vector<unsigned int> FKDTree<TYPE, numberOfDimensions>::search_in_the_box_linear(const FKDPoint<TYPE, numberOfDimensions>& minPoint, const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
	std::vector<unsigned int> indecesToVisit;
	std::vector<unsigned int> result;
	result.reserve(16);
	indecesToVisit.push_back(0);
	unsigned int index = 0;
	unsigned int dimension = 0;
	unsigned int depth = 0;
	while (! indecesToVisit.empty()) {
		index = indecesToVisit.back();
		indecesToVisit.pop_back();
		depth = FLOOR_LOG2(index + 1);
		dimension = depth % numberOfDimensions;
		bool intersection = intersects(index, minPoint, maxPoint, dimension);
		if (intersection && isInTheBox(index, minPoint, maxPoint))
			result.push_back(theIds[index]);
		bool isLowerThanBoxMin = theDimensions[dimension][index] < minPoint[dimension];
		int startSon = isLowerThanBoxMin; //left son = 0, right son =1
		int endSon = isLowerThanBoxMin || intersection;
		for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
			unsigned int indexToAdd = leftSonIndex(index) + whichSon;
			if (indexToAdd < theNumberOfPoints)
				indecesToVisit.push_back(indexToAdd);
		}
	}
	return result;
}

template<class TYPE, int numberOfDimensions>
void FKDTree<TYPE, numberOfDimensions>::build() {
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
			std::nth_element(thePoints.begin() + theIntervalMin[indexInArray], thePoints.begin() + theIntervalMin[indexInArray] + whichElementInInterval, thePoints.begin() + theIntervalMin[indexInArray] + theIntervalLength[indexInArray],
					[dimension](const FKDPoint<TYPE,numberOfDimensions> & a, const FKDPoint<TYPE,numberOfDimensions> & b) -> bool
					{
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
	std::cout << "[FKDTree] Node set in: " << node_find_time << " ms." <<  std::endl;
	dimension = theDepth % numberOfDimensions;
	unsigned int firstIndexInDepth = (1 << theDepth) - 1;
	for (unsigned int indexInArray = firstIndexInDepth; indexInArray < theNumberOfPoints; ++indexInArray)
		add_at_position(thePoints[theIntervalMin[indexInArray]], indexInArray);
}

template<class TYPE, int numberOfDimensions>

unsigned int FKDTree<TYPE, numberOfDimensions>::leftSonIndex(unsigned int index) const {
	return 2 * index + 1;
}

template<class TYPE, int numberOfDimensions>

unsigned int FKDTree<TYPE, numberOfDimensions>::rightSonIndex(unsigned int index) const {
	return 2 * index + 2;
}

template<class TYPE, int numberOfDimensions>

bool FKDTree<TYPE, numberOfDimensions>::intersects(unsigned int index, const FKDPoint<TYPE, numberOfDimensions>& minPoint, const FKDPoint<TYPE, numberOfDimensions>& maxPoint, int dimension) const {
	return (theDimensions[dimension][index] <= maxPoint[dimension] && theDimensions[dimension][index] >= minPoint[dimension]);
}

template<class TYPE, int numberOfDimensions>

bool FKDTree<TYPE, numberOfDimensions>::isInTheBox(unsigned int index, const FKDPoint<TYPE, numberOfDimensions>& minPoint, const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
	bool inTheBox = true;
#pragma unroll
	for (int i = 0; i < numberOfDimensions; ++i) {
		inTheBox &= (theDimensions[i][index] <= maxPoint[i] && theDimensions[i][index] >= minPoint[i]);
	}
	return inTheBox;
}

#endif /* FKDTREE_FKDTREE_H_ */
