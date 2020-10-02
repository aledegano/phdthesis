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

int main() {

	const long int num_points = 20;
	float range = 2.0;
	const unsigned int maxNN = ceil(pow(2 * range, 3) / pow(100, 3) * num_points * 2.0);

	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);
	std::vector<std::vector<unsigned int>> fkd_results;
	std::vector<std::vector<unsigned int>> cuda_results(num_points);
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

	std::cout << "Evaluating performance for " << num_points << " points in range " << range << " and Max Results for GPU " << maxNN << "." << std::endl;

	std::chrono::time_point<std::chrono::system_clock> start, end, start2, end2;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	std::cout << "Running on cuda device: " << devProp.name << std::endl;

	start = std::chrono::system_clock::now();
	CudaFKDTree cudatree(num_points, cuda_points);
	cudatree.build();
	end = std::chrono::system_clock::now();
	float cuda_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "[CudaFKDTree] Tree built in " << cuda_build_time << " ms." << std::endl;
	if (!cudatree.test_correct_build()) {
		std::cout << "[CudaFKDTree] ERROR! Build wrong!" << std::endl;
		return 1;
	}

	start = std::chrono::system_clock::now();
//	cudatree.search_in_the_box_linear(search_box, maxNN, arr_cuda_results);
	cuda_results = cudatree.search_in_the_box_sequential(search_box);
	end = std::chrono::system_clock::now();
	float cuda_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "[CudaFKDTree] Search completed in: " << cuda_search_time << " ms. Total: " << cuda_build_time + cuda_search_time << " ms." << std::endl;
//	unsigned int thisSize = 0;
//	for (unsigned int i = 0; i < num_points; ++i) {
//		thisSize = arr_cuda_results[i * maxNN];
//		if (thisSize > 0) {
//			cuda_results[i].resize(thisSize);
//			for (unsigned int j = 1; j <= thisSize; ++j)
//				cuda_results[i][j - 1] = arr_cuda_results[i * maxNN + j];
//		}
//	}

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
	start = std::chrono::system_clock::now();
	for (long int i = 0; i < num_points; ++i)
		fkd_results.push_back(tree2.search_in_the_box_linear(cpu_search_box[2 * i], cpu_search_box[2 * i + 1]));
	end = std::chrono::system_clock::now();
	float cpu_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "[FKDTree-linear] Search completed in: " << cpu_search_time << " ms. Total: " << cpu_build_time + cpu_search_time << " ms." << std::endl;

	std::cout << "Starting comparison." << std::endl;
	unsigned int samples, steps;
	if (num_points > 20) {
		samples = 100;
		steps = num_points / samples;
	} else {
		samples = num_points;
		steps = 1;
	}
	unsigned int id = 0;
	unsigned int size_check = 0;
	unsigned int nn_check = 0;
	for (unsigned int samp = 0; samp < samples; ++samp) {
		id = samp * steps;
		std::sort(fkd_results[id].begin(), fkd_results[id].end());
		std::sort(cuda_results[id].begin(), cuda_results[id].end());
		//Check number of results
		if (fkd_results[id].size() != cuda_results[id].size()) {
			std::cout << std::endl << "Error different size: " << std::endl;
			std::cout << "[FKDTree]: " << fkd_results[id].size() << std::endl;
			std::cout << "[CudaKDTree]: " << cuda_results[id].size() << std::endl;
			std::cout << "Point: ";
			points[id].print();
			std::cout << "[FKDTree]: " << std::endl;
			for (auto point : fkd_results[id])
				points[point].print();
			std::cout << "[CudaKDTree]: " << std::endl;
			for (auto point : cuda_results[id])
				points[point].print();
			continue;
		} else
			++size_check;
		for (size_t i = 0; i < fkd_results[id].size(); ++i) {
			if (fkd_results[id][i] != cuda_results[id][i]) {
				std::cout << "Error different result: " << std::endl;
				std::cout << "Point searched for: ";
				points[id].print();
				std::cout << "[FKDTree] NN: ";
				points[fkd_results[id][i]].print();
				std::cout << "[CudaKDTree] NN: ";
				points[cuda_results[id][i]].print();
			} else
				++nn_check;
		}
	}
	std::cout << "Comparison completed." << std::endl;
	std::cout << "Number of points with matching NN size: " << size_check << std::endl;
	std::cout << "Total number of matching NN: " << nn_check << std::endl;
}
