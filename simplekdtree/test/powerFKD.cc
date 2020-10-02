#include "CudaKDtree/KDTree/src/FKDTree.h"
#include "CudaKDtree/KDTree/src/CudaFKDTree.cuh"

#include <iostream>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <bits/random.h>
#include <memory>
#include <chrono>
#include <algorithm>

int main(int argc, char** argv) {

	if (argc != 3) {
		std::cerr << "Two arguments are required: 1 for CPU search, 2 for GPU search. Followed by number of repetitions." << std::endl;
		return 1;
	}
	unsigned int searchType = 0;
	unsigned int repeat = atoi(argv[2]);
	searchType = atoi(argv[1]);

	const long int num_points = 500000;
	float range = 2.0;
	const unsigned int maxNN = ceil(pow(2 * range, 3) / pow(100, 3) * num_points * 2.0);

	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);
	unsigned int* arr_cuda_results;
	CudaSafeCall(cudaMallocHost((void** )&arr_cuda_results, num_points * maxNN * sizeof(unsigned int)));
	std::vector<CudaFKDPoint> cuda_points;
	std::vector<FKDPoint<float, 3> > cpu_search_box;
	std::vector<FKDPoint<float, 3> > points;
	cpu_search_box.resize(2 * num_points);

	points.resize(num_points);
	cuda_points.resize(num_points);

	CudaFKDPoint* search_box;
	CudaSafeCall(cudaMallocHost((void** )&search_box, num_points * 2 * sizeof(CudaFKDPoint)));

	for (unsigned i = 0; i < num_points; i++) {
		float x = rnd(generator);
		float y = rnd(generator);
		float z = rnd(generator);
		points[i] = FKDPoint<float, 3>(x, y, z, i);
		cuda_points[i] = CudaFKDPoint(x, y, z, i);
		for (unsigned j = 0; j < 3; ++j) {
			search_box[2 * i][j] = cuda_points[i][j] - range;
			cpu_search_box[2 * i][j] = search_box[2 * i][j];
			search_box[2 * i + 1][j] = cuda_points[i][j] + range;
			cpu_search_box[2 * i + 1][j] = search_box[2 * i + 1][j];
		}
	}

//	std::cout << "Evaluating performance for " << num_points << " points in range " << range << " and Max Results for GPU " << maxNN << "." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> start, end, start2, end2;

	if (searchType == 2) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		std::cout << "Running on cuda device: " << devProp.name << std::endl;
		start = std::chrono::system_clock::now();
		CudaFKDTree cudatree(num_points, cuda_points);
		cudatree.build();
		end = std::chrono::system_clock::now();
//		float cuda_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//		std::cout << "[CudaFKDTree] Tree built in " << cuda_build_time << " ms." << std::endl;
		if (!cudatree.test_correct_build()) {
			std::cout << "[CudaFKDTree] ERROR! Build wrong!" << std::endl;
			return 1;
		}

//		std::cout << "[CudaFKDTree] Repeating the search: " << repeat << " times." << std::endl;
		start = std::chrono::system_clock::now();
//		for (unsigned int i = 0; i < repeat; i++) {
//			if (i % 1000 == 0)
//				std::cout << "Repetition: " << i << std::endl;
//			cudatree.search_in_the_box_linear(search_box, maxNN, arr_cuda_results, 1 , 32, repeat);
//		}
		end = std::chrono::system_clock::now();
//		float cuda_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//		std::cout << "[CudaFKDTree] Search completed in: " << cuda_search_time << " ms. Total: " << cuda_build_time + cuda_search_time << " ms." << std::endl;
	} else {
		start = std::chrono::system_clock::now();
		FKDTree<float, 3> tree2(num_points, points);
		tree2.build();
		end = std::chrono::system_clock::now();
		float cpu_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[FKDTree] Tree built in " << cpu_build_time << " ms." << std::endl;
		if (!tree2.test_correct_build()) {
			std::cout << "[FKDTree] ERROR! Build wrong!" << std::endl;
			return 1;
		}

//		std::cout << "[FKDTree] Repeating the search: " << repeat << " times." << std::endl;
		start = std::chrono::system_clock::now();
		for (unsigned int i = 0; i < repeat; i++) {
			for (long int i = 0; i < num_points; ++i)
				tree2.search_in_the_box_linear(cpu_search_box[2 * i], cpu_search_box[2 * i + 1]);
		}
		end = std::chrono::system_clock::now();
		float cpu_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "[FKDTree CPU] " << cpu_search_time << " " << repeat << std::endl;
//		std::cout << "[FKDTree-linear] Search completed in: " << cpu_search_time << " ms. Total: " << cpu_build_time + cpu_search_time << " ms." << std::endl;
	}
}
