#include "CudaKDtree/KDTree/src/kdtree.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <bits/random.h>
#include <sstream>
#include <math.h>
#include <vector>

std::string print_float4(float4 print_me) {
	std::ostringstream string;
	string << print_me.x << ", " << print_me.y << ", " << print_me.z << ". ";
	return string.str();
}

int main(int argc, char** argv) {
	if (argc != 4) {
		std::cerr << "This program requires exactly 3 argument: the number of points to generate, the range to search and the max NN to search." << std::endl;
		return 1;
	}
	unsigned int num_points = atoi(argv[1]);
	float range_input = atoi(argv[2]);
	float4 points[num_points];
	float4 range[num_points];
	unsigned int max_nn = atoi(argv[3]);
	constexpr int threads_per_block = 1024;

	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);

	for (unsigned int i = 0; i < num_points; i++) {
		points[i].x = rnd(generator);
		points[i].y = rnd(generator);
		points[i].z = rnd(generator);
		points[i].w = i; //Use this to save the index
		range[i] = make_float4(range_input, range_input, range_input, i);
	}

	std::cout << "Build the tree on CPU for " << num_points << " points. And search for " << max_nn << " nearest neighbors." << std::endl;
	kdtree tree(points, num_points);
	std::cout << "Tree built with " << tree.getMaxDepth() << " layers." << std::endl;
	std::cout << std::endl;

	std::cout << "The results will be allocated in an array of " << num_points * max_nn * sizeof(int) / 1024 << " kb." << std::endl;
	size_t stackSizePerThread = 4096;
	cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetLimit(cudaLimitStackSize, stackSizePerThread);
	float milliseconds = tree.AllNNKernel(range, max_nn, threads_per_block);
//	float milliseconds = tree.AllNNSeq(range, z_thresh, max_nn);
	std::cout << "GPU all NN find complete, took " << milliseconds + tree.copy_points_cost << " ms." << std::endl;
	std::cout << std::endl;

//	for (unsigned int i = 0; i < num_points; i++) {
//		std::cout << "NNs of point " << i << std::endl;
//		for (unsigned int j = 0; j < max_nn; j++)
//			std::cout << tree.getAllNN(i, j) << " ";
//		std::cout << std::endl;
//	}

	int* naive_res = new int[num_points * max_nn];
	for (unsigned int i = 0; i < num_points * max_nn; i++)
		naive_res[i] = -1;
	std::cout << "Begin naive method" << std::endl;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	unsigned int cand_id = 0;
	for (unsigned int i = 0; i < num_points; i++) {
		cand_id = 0;
		for (unsigned int j = 0; j < num_points; j++) {
			if (j == i)
				continue;
			if (points[i] >= points[j] - range[i] and points[i] <= points[j] + range[i]) {
				if (cand_id <= max_nn) {
					naive_res[cand_id + max_nn * i] = j;
					cand_id++;
				}
			}
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Naive NN find complete, took " << milliseconds << " ms." << std::endl;

	std::vector<float> gpu_dist;
	std::vector<float> naive_dist;
	int gpu_missing = 0;
	int naive_missing = 0;
	int naive_better = 0;
	int gpu_better = 0;
	std::cout << "Check results: " << std::endl;
	for (unsigned int i = 0; i < num_points; i++) {
		gpu_dist.clear();
		naive_dist.clear();
		bool goOn = true;
		int result = 0;
		unsigned int j = 0;
		while (goOn) {
			if (j > max_nn) {
				goOn = false;
				break;
			}
			result = tree.getAllNN(i, j);
			if (result < 0) {
				goOn = false;
			} else {
				gpu_dist.push_back(sqrt(squared_distance(points[i], points[result])));
				j++;
			}
		}
		std::sort(gpu_dist.begin(), gpu_dist.end());
		goOn = true;
		result = 0;
		j = 0;
//		if (i % 100 == 0)
//			std::cout << "Gpu first NN: " << gpu_dist[0] << std::endl;
		while (goOn) {
			if (j > max_nn) {
				goOn = false;
				break;
			}
			result = naive_res[j + i * max_nn];
			if (result < 0) {
				goOn = false;
			} else {
				naive_dist.push_back(sqrt(squared_distance(points[i], points[result])));
				j++;
			}
		}
		std::sort(naive_dist.begin(), naive_dist.end());
//		if (i % 100 == 0)
//			std::cout << "Naive first NN: " << naive_dist[0] << std::endl;
		if (gpu_dist.size() < naive_dist.size()) {
			gpu_missing++;
		} else if (gpu_dist.size() > naive_dist.size()) {
			naive_missing++;
		} else {
			for (unsigned int k = 0; k < gpu_dist.size(); k++) {
				if (naive_dist[k] < gpu_dist[k]) {
					naive_better++;
				} else if (naive_dist[k] > gpu_dist[k]) {
					gpu_better++;
				}
			}
		}
	}
	std::cout << "Report: \n" << "Gpu missed NN: " << gpu_missing << "\n Naive missed NN: " << naive_missing << std::endl;
	std::cout << "Gpu better NN: " << gpu_better << "\n Naive better NN: " << naive_better << std::endl;

	delete naive_res;
}

