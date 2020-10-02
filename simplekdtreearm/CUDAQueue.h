/*
 * CUDAQueue.h
 *
 *  Created on: Jun 17, 2015
 *      Author: fpantale
 */

#ifndef INCLUDE_CUDAQUEUE_H_
#define INCLUDE_CUDAQUEUE_H_

#include <cuda_runtime.h>
#include <host_defines.h>
#include <cassert>

// CUDAQueue is a single-block queue.
// One may want to use it as a __shared__ struct, and have multiple threads
// pushing data into it.
template<int maxSize, class T>
struct CUDAQueue {
	__forceinline__ __host__ __device__ CUDAQueue() :
			m_size(0) {
	}

	__inline__ __device__
	int push(const T& element) {
#ifdef __CUDACC__
		auto previousSize = atomicAdd(&m_size, 1);
		if(previousSize < maxSize) {
			m_data[previousSize] = element;
			return previousSize;
		} else
		atomicSub(&m_size, 1);
#endif
		return -1;
	}
	;

	__inline__ __host__ __device__
	int push_singleThread(const T& element) {
		auto previousSize = m_size++;
		if (previousSize < maxSize) {
			m_data[previousSize] = element;
			return previousSize;
		} else
			return -1;
	}
	;


	__inline__ __device__
	T pop_back() {
#ifdef __CUDACC__
		assert(m_size > 0);
		auto previousSize = atomicAdd (&m_size, -1);
		return m_data[previousSize];
#endif
	};


	__inline__  __host__   __device__
	T pop_back_singleThread() {
		assert(m_size > 0);
		auto previousSize = m_size--;
		return m_data[previousSize - 1];
	};

	__inline__  __host__   __device__ T back() {
		assert(m_size > 0);
		return m_data[m_size - 1];
	};

	__inline__ __host__ __device__
	void reset() {
		m_size = 0;
	};

	T m_data[maxSize];
	int m_size;
};

#endif /* INCLUDE_CUDAQUEUE_H_ */
