#include "CudaKDtree/KDTree/src/kdtree.h"
#include <iostream>
#include <random>
#include <bits/random.h>
#include <memory>

int main(){
//	std::random_device rd;
	std::default_random_engine generator(42); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);
	unsigned num_points = 100000;
	std::cout << "Instantiating list of " << num_points << " float4 ranges on stack." << std::endl;
	std::unique_ptr<float4[]> ranges (new float4[num_points]);
	std::cout << "Instantiating list of " << num_points << " float4 points on stack." << std::endl;
	std::unique_ptr<float4[]> points (new float4[num_points]);
	std::cout << "Assigning random values to points and const value to ranges." << std::endl;
	float range = 1.1;
	unsigned max_nn = 2000;
	for(unsigned i = 0; i<num_points; i++){
		points[i] = make_float4(rnd(generator),rnd(generator),rnd(generator),i);
		ranges[i] = make_float4(range, range, range, i);
	}

	std::cout << "Instantiating and creating a new kdtree on stack." << std::endl;
	std::unique_ptr<kdtree> tree (new kdtree(points.get(), num_points));
	std::cout << "NN search completed in: " << tree->AllNNSeq(ranges.get(), max_nn) << " ms." << std::endl;
	return 0;
}
