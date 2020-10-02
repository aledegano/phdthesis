#include "CudaKDtree/KDTree/src/FKDTree.h"
#include <iostream>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <bits/random.h>
#include <CudaKDtree/KDTree/src/CudaFKDTree.cuh>
#include <memory>
#include <chrono>
#include <algorithm>

std::vector<unsigned int> naive_search(std::vector<FKDPoint<float, 3> > points, FKDPoint<float, 3> search_box_min, FKDPoint<float, 3> search_box_max) {
	std::vector<unsigned int> result;
	for (auto& point : points) {
		bool inside = true;
		for (unsigned int i = 0; i < 3; i++)
			inside *= (point[i] >= search_box_min[i] && point[i] <= search_box_max[i]);
		if (inside)
			result.push_back(point.getId());
	}
	return result;
}

int main(){
	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);
	const long int num_points = 300000;
	std::vector<FKDPoint<float, 3> > points;
	std::vector<FKDPoint<float, 3> > search_box;
	std::vector<std::vector<unsigned int> > results;
	results.resize(num_points);
	points.resize(num_points);
	search_box.resize(2*num_points);
	float range = 2.1;
	for (unsigned i = 0; i < num_points; i++) {
		float x = rnd(generator);
		float y = rnd(generator);
		float z = rnd(generator);
		points[i] = FKDPoint<float, 3>(x, y, z, i);
		for (unsigned j = 0; j < 3; ++j) {
			search_box[2 * i][j] = points[i][j] - range;
			search_box[2 * i + 1][j] = points[i][j] + range;
		}
	}
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::cout << "Evaluating performance for " << num_points << " points in range " << range << "." << std::endl;

	start = std::chrono::system_clock::now();
	FKDTree<float, 3> tree(num_points, points);
	tree.build();
	for (unsigned i = 0; i < num_points; i++)
		results[i] = tree.search_in_the_box_linear(search_box[2 * i], search_box[2 * i + 1]);
	end = std::chrono::system_clock::now();
	float cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "[FKDTree-linear] Search completed in: " << cpu_time<< " ms." << std::endl;

	int samples = 1000;
	int step = num_points/samples;
	int id = 0;
	std::vector<unsigned int> naive_res;
	int sizeErr = 0;
	int rightSize = 0;
	int nnErr = 0;
	int rightNN = 0;
	for(int i=0; i<samples; i++){
		id = step*i;
		naive_res = naive_search(points, search_box[2 * id], search_box[2 * id + 1] );
		if(naive_res.size() != results[id].size()){
			std::cout << "Size error for id: " << id << " naive= " << naive_res.size() << " tree= " << results[id].size() << std::endl;
			sizeErr++;
			continue;
		} else
			rightSize++;
		std::sort(naive_res.begin(), naive_res.end());
		std::sort(results[id].begin(), results[id].end());
		for(unsigned int j=0; j<naive_res.size(); j++){
			if(naive_res[j] != results[id][j]){
				std::cout << "Different NN result for point: ";
				points[id].print();
				std::cout << " naive found: ";
				points[naive_res[j]].print();
				std::cout << " FKDTree found: ";
				points[results[id][j]].print();
				nnErr++;
				continue;
			} else
				rightNN++;
		}
	}
	std::cout << "####################################################" << std::endl;
	std::cout << "Unit test completed." << std::endl;
	std::cout << "####################################################" << std::endl;
	std::cout << "Different sizes: " << sizeErr << " different NN: " << nnErr << std::endl;
	std::cout << "Correct sizes: " << rightSize << " correct NN: " << rightNN << std::endl;
	return 0;
}
