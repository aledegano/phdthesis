#include "CudaKDtree/KDTree/src/kdtree.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <bits/random.h>
#include <sstream>
#include <math.h>
#include <vector>
#include <memory>
#include "tbb/parallel_sort.h"
#include <sstream>
#include <iomanip>

std::string print_float4(float4 print_me) {
	std::ostringstream string;
	string << std::setw(2) << std::setprecision(2) << print_me.w << ": " << std::setw(4) << std::setprecision(4) << print_me.x << ", " << std::setw(4) << std::setprecision(4) << print_me.y << ", " << std::setw(4) << std::setprecision(4)
			<< print_me.z << ". ";
	return string.str();
}

int main(){
	std::default_random_engine generator(42); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);
	unsigned num_points = 20;
//	std::unique_ptr<float4[]> ranges (new float4[num_points]);
	float4* ranges = new float4[num_points];
	std::unique_ptr<float4[]> points (new float4[num_points]);
	std::vector<int> tmpRes;
	std::vector<std::vector<int> > treeResults;
	std::vector<std::vector<int> > naiveResults;
	float range = 20;
	unsigned max_nn = num_points;
	for(unsigned i = 0; i<num_points; i++){
		points[i] = make_float4(ceil(rnd(generator)),ceil(rnd(generator)),ceil(rnd(generator)),i);
		ranges[i] = make_float4(range, range, range, i);
	}
	std::cout << "Instantiating and creating a new kdtree on stack." << std::endl;
	std::unique_ptr<kdtree> tree (new kdtree(points.get(), num_points));
	std::cout << "Performing NN tree search." << std::endl;
	tree->AllNNTbbPar(ranges, max_nn);
	for(unsigned i=0; i<num_points; i++){
		std::cout << "Tree point: " << print_float4(points[i]) << std::endl;
		for(unsigned j=0; j<max_nn; j++){
			tmpRes.push_back(tree->getAllNN(i, j));
			if(tmpRes.back() == -1){
				break;
			} else {
				std::cout << "Tree NN: " << print_float4(points[tmpRes.back()]) << std::endl;
			}
		}
		treeResults.push_back(tmpRes);
		tmpRes.clear();
	}
	std::cout << "Performing NN naive search." << std::endl;
	tree->AllNNNaive(ranges, max_nn);
	for(unsigned i=0; i<num_points; i++){
		std::cout << "Naive point: " << print_float4(points[i]) << std::endl;
		for(unsigned j=0; j<max_nn; j++){
			tmpRes.push_back(tree->getAllNN(i, j));
			if(tmpRes.back() == -1){
				break;
			} else {
				std::cout << "Naive NN: " << print_float4(points[tmpRes.back()]) << std::endl;
			}
		}
		naiveResults.push_back(tmpRes);
		tmpRes.clear();
	}
	std::cout << "#####################################" << std::endl;
	std::cout << "Analyzing differences: " << std::endl;
	std::cout << "#####################################" << std::endl;
	for(unsigned i=0; i<num_points; i++){
		std::cout << "Point: " << i << " naive size: " << naiveResults[i].size() << " tree size: " << treeResults[i].size() << std::endl;
	}

}
