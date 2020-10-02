/*
 * CudaFFKDPoint.cuh
 *
 *  Created on: Mar 8, 2016
 *      Author: degano
 */

#ifndef CUDAFKDTREE_KDPOINT_H_
#define CUDAFKDTREE_KDPOINT_H_

#include <cuda.h>
#include <host_defines.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <iostream>
#include <assert.h>
#include <cstring>

static constexpr unsigned int numberOfDimensions = 3;

class CudaFKDPoint {

public:
	__host__ __device__
	CudaFKDPoint() :
			theElements(), theId(0) {
	}
	__host__ __device__
	CudaFKDPoint(const CudaFKDPoint& other) :
			theId(other.theId) {
		for (unsigned int i = 0; i < numberOfDimensions; ++i)
			theElements[i] = other.theElements[i];
	}

	__host__ __device__
	CudaFKDPoint operator=(const CudaFKDPoint other) {
		if (this != &other) {
			theId = other.theId;
			for (unsigned int i = 0; i < numberOfDimensions; ++i)
				theElements[i] = other.theElements[i];
		}
		return *this;
	}

	__host__ __device__
	CudaFKDPoint(float x, float y, float z, unsigned int id) {
		static_assert( numberOfDimensions == 3, "Point dimensionality differs from the number of passed arguments." );
		theId = id;
		theElements[0] = x;
		theElements[1] = y;
		theElements[2] = z;
	}

	__host__ __device__
	float& operator[](unsigned int const i) {
//		assert(i < numberOfDimensions);
		return theElements[i];
	}
	__host__ __device__
	float const& operator[](unsigned int const i) const {
//		assert(i < numberOfDimensions);
		return theElements[i];
	}
	__host__ __device__
	void setDimension(unsigned int i, const float& value) {
		assert(i < numberOfDimensions);
		theElements[i] = value;
	}
	__host__ __device__
	void setId(const unsigned int id) {
		theId = id;
	}
	__host__ __device__
	unsigned int getId() const {
		return theId;
	}
	void print() {
		std::cout << "point id: " << theId << std::endl;
		for (unsigned i = 0; i < numberOfDimensions; ++i) {
			std::cout << theElements[i] << " ";
		}
		std::cout << std::endl;
	}

private:
	float theElements[numberOfDimensions];
	unsigned int theId;
};

#endif /* CUDAFKDTREE_KDPOINT_H_ */
