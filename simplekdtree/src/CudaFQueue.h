/*
 * CudaFQueue.h
 *
 *  Created on: Mar 8, 2016
 *      Author: degano
 */

#ifndef CUDAFKDTREE_QUEUE_H_
#define CUDAFKDTREE_QUEUE_H_
#include <vector>

class CudaFQueue {
public:
	CudaFQueue();
	CudaFQueue(unsigned int capacity);
	CudaFQueue(const CudaFQueue & v);
	CudaFQueue(CudaFQueue && other) :
			theSize(0), theFront(0), theTail(0) {
		theBuffer.clear();
		theSize = other.theSize;
		theFront = other.theFront;
		theTail = other.theTail;
		theBuffer = other.theBuffer;
		other.theSize = 0;
		other.theFront = 0;
		other.theTail = 0;
	}

	CudaFQueue& operator=(CudaFQueue && other) {
		if (this != &other) {
			theBuffer.clear();
			theSize = other.theSize;
			theFront = other.theFront;
			theTail = other.theTail;
			theBuffer = other.theBuffer;
			other.theSize = 0;
			other.theFront = 0;
			other.theTail = 0;
		}
		return *this;
	}
	~CudaFQueue();

	unsigned int capacity() const;
	const unsigned int* data() const;
	unsigned int size() const;
	bool empty() const;
	float front();
	float tail();
	void push_back(const float value);
	void pop_front();
	void pop_front(const unsigned int numberOfElementsToPop);

	void reserve(unsigned int capacity);
	void resize(unsigned int capacity);

	float operator[](unsigned int index);
	CudaFQueue & operator=(const CudaFQueue &);
	void clear();
private:
	unsigned int theSize;
	unsigned int theFront;
	unsigned int theTail;
	std::vector<unsigned int> theBuffer;

};
// Your code goes here ...

CudaFQueue::CudaFQueue() {
	theSize = 0;
	theBuffer.resize(0);
	theFront = 0;
	theTail = 0;

}

CudaFQueue::CudaFQueue(const CudaFQueue & v) {
	theSize = v.theSize;
	theBuffer = v.theBuffer;
	theFront = v.theFront;
	theTail = v.theTail;
}

CudaFQueue::CudaFQueue(unsigned int capacity) {
	theBuffer.resize(capacity);
	theSize = 0;
	theFront = 0;
	theTail = 0;
}

CudaFQueue & CudaFQueue::operator =(const CudaFQueue & v) {
	if (this != &v) {
		theBuffer.clear();
		theSize = v.theSize;
		theBuffer = v.theBuffer;
		theFront = v.theFront;
		theTail = v.theTail;
	}
	return *this;

}

float CudaFQueue::front() {
	return theBuffer[theFront];
}

float CudaFQueue::tail() {
	return theBuffer[theTail];
}

void CudaFQueue::push_back(const float v) {
//	std::cout << "head tail and size before pushing " << theFront << " " << theTail << " " << theSize << std::endl;
//	std::cout << "content before pushing" << std::endl;
//	for(int i =0; i< theSize; i++)
//		std::cout << theBuffer.at((theFront+i)%theBuffer.capacity()) << std::endl;
	if (theSize >= theBuffer.size()) {
		auto oldCapacity = theBuffer.size();
		theBuffer.reserve(oldCapacity + theTail);

		if (theFront != 0) {
//			std::copy(theBuffer.begin(), theBuffer.begin() + theTail, theBuffer.begin() + oldCapacity);
			for (unsigned int i = 0; i < theTail; ++i) {
				theBuffer.push_back(theBuffer[i]);
			}
			theTail = 0;

		} else {
			theBuffer.resize(oldCapacity + 16);
			theTail += oldCapacity;
		}
//		theTail += oldCapacity;

//		std::cout << "resized" << std::endl;
	}

	theBuffer[theTail] = v;
	theTail = (theTail + 1) % theBuffer.size();
	theSize++;
//	std::cout << "head and tail after pushing " << theFront << " " << theTail << " " << theSize << std::endl;
//
//	std::cout << "content after pushing" << std::endl;
//	for(int i =0; i< theSize; i++)
//		std::cout << theBuffer.at((theFront+i)%theBuffer.capacity()) << std::endl;
//	std::cout << "\n\n" << std::endl;

}

void CudaFQueue::pop_front() {
	if (theSize > 0) {
		theFront = (theFront + 1) % theBuffer.size();
		theSize--;
	}
}

void CudaFQueue::reserve(unsigned int capacity) {
	theBuffer.reserve(capacity);
}

unsigned int CudaFQueue::size() const {
	return theSize;
}

void CudaFQueue::resize(unsigned int capacity) {
	theBuffer.resize(capacity);
}

float CudaFQueue::operator[](unsigned int index) {
	return theBuffer[(theFront + index) % theBuffer.size()];
}

unsigned int CudaFQueue::capacity() const {
	return theBuffer.capacity();
}

const unsigned int* CudaFQueue::data() const {
	return theBuffer.data();
}

CudaFQueue::~CudaFQueue() {
}

void CudaFQueue::clear() {
	theBuffer.clear();
	theSize = 0;
	theFront = 0;
	theTail = 0;
}

void CudaFQueue::pop_front(const unsigned int numberOfElementsToPop) {
	unsigned int elementsToErase = theSize > numberOfElementsToPop ? numberOfElementsToPop : theSize;
	theSize -= elementsToErase;
	theFront = (theFront + elementsToErase) % theBuffer.size();
}

#endif
