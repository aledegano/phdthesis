#include "CudaKDtree/KDTree/src/kdtree.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <bits/random.h>
#include <sstream>
#include <math.h>
#include <vector>
#include "tbb/parallel_sort.h"
#include <iomanip>

std::string print_float4(float4 print_me) {
	std::ostringstream string;
	string << std::setw(4) << std::setprecision(4) << print_me.x << ", " << std::setw(4) << std::setprecision(4) << print_me.y << ", " << std::setw(4) << std::setprecision(4) << print_me.z << ". ";
	return string.str();
}

int main(int argc, char** argv) {
	int expect_args = 3;
	if (argc != expect_args + 1) {
		std::cerr << "This program requires exactly " << expect_args << " argument: the number of points to generate, the range to search the NN and the sample to analyze." << std::endl;
		return 1;
	}
	unsigned int num_points = atoi(argv[1]);
	float range_input = atoi(argv[2]);
	int sample = atoi(argv[3]);
	float4 points[num_points];
	float4 range[num_points];
	unsigned int max_nn = num_points;
	std::cout << "For the unit test all the NN of each point will be searched." << std::endl;

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

	size_t stackSizePerThread = 4096;
	cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetLimit(cudaLimitStackSize, stackSizePerThread);

	float milliseconds = tree.AllNNKernel(range, max_nn, 1024);
//	float milliseconds = tree.AllNNTbbPar(range, max_nn);
	std::cout << "GPU all NN find complete, took " << milliseconds << " ms." << std::endl;
	std::cout << std::endl;

	std::vector<std::vector<int> > gpu_results;
	std::vector<int> tmp;
	for (int i = 0; i < (int) ceil(num_points / sample); i++) {
		for (unsigned j = 0; j < max_nn; j++) {
			tmp.push_back(tree.getAllNN(i * sample, j));
			if (tree.getAllNN(i * sample, j) == -1)
				break;
		}
		tbb::parallel_sort(tmp.begin(), tmp.end(), std::greater<int>()); // Sort by index, only to compare easily between methods
		gpu_results.push_back(tmp);
		tmp.clear();
	}

	std::cout << "Beginning Naive search" << std::endl;
	float naive_time = tree.AllNNNaive(range, max_nn);
	std::cout << "Naive all NN find complete, took " << naive_time << " ms." << std::endl;
	std::cout << std::endl;

	std::vector<std::vector<int> > naive_results;
	for (int i = 0; i < (int) ceil(num_points / sample); i++) {
		for (uint j = 0; j < max_nn; j++) {
			tmp.push_back(tree.getAllNN(i * sample, j));
			if (tree.getAllNN(i * sample, j) == -1)
				break;
		}
		tbb::parallel_sort(tmp.begin(), tmp.end(), std::greater<int>()); // Sort by index, only to compare easily between methods
		naive_results.push_back(tmp);
		tmp.clear();
	}

	int gpu_not_found = 0;
	int naive_not_found = 0;
	int different = 0;
	int analyzed = 0;
	std::vector<float> d1;
	std::vector<float> d2;
	std::vector<float> delta_closure;
	std::vector<float4> different_points;

	std::vector<float4> gpu_miss_point;
	std::vector<float4> gpu_miss_naiveNN;
	std::vector<float4> naive_miss_point;
	std::vector<float4> naive_miss_gpuNN;
	int this_point_has_diff = 0;

	for (int i = 0; i < (int) ceil(num_points / sample); i++) {
		this_point_has_diff = 0;
		for (uint j = 0; j < std::min(gpu_results[i].size(), naive_results[i].size()); j++) {
			analyzed++;
			if (gpu_results.at(i).at(j) != naive_results.at(i).at(j)) {
				if (gpu_results.at(i).at(j) == -1) {
					gpu_not_found++;
					float4 point = points[i * sample];
					float4 naive_NN = points[naive_results.at(i).at(j)];
					gpu_miss_point.push_back(point);
					gpu_miss_naiveNN.push_back(naive_NN);
					break; // There are no more valid value after the trailing -1
				} else if (naive_results.at(i).at(j) == -1) {
					naive_not_found++;
					float4 point = points[i * sample];
					float4 gpu_NN = points[gpu_results.at(i).at(j)];
					naive_miss_point.push_back(point);
					naive_miss_gpuNN.push_back(gpu_NN);
					break; // There are no more valid value after the trailing -1
				} else {
					// Don't count twice if this two NN are different only because are shifted
					if ((gpu_results.at(i).at(j) == naive_results.at(i).at(j - this_point_has_diff) or gpu_results.at(i).at(j - this_point_has_diff) == naive_results.at(i).at(j)) and this_point_has_diff > 0)
						continue;
					this_point_has_diff++;
					different++;
					float4 point = make_float4(points[i * sample].x, points[i * sample].y, points[i * sample].z, i * sample);
					float4 gpu_NN = make_float4(points[gpu_results.at(i).at(j)].x, points[gpu_results.at(i).at(j)].y, points[gpu_results.at(i).at(j)].z, gpu_results.at(i).at(j));
					float4 naive_NN = make_float4(points[naive_results.at(i).at(j)].x, points[naive_results.at(i).at(j)].y, points[naive_results.at(i).at(j)].z, naive_results.at(i).at(j));
					d1.push_back(sqrt(squared_distance(point, gpu_NN)));
					d2.push_back(sqrt(squared_distance(point, naive_NN)));
					different_points.push_back(point);
					different_points.push_back(gpu_NN);
					different_points.push_back(naive_NN);
				}
			} else {
				float4 point = make_float4(points[i * sample].x, points[i * sample].y, points[i * sample].z, i * sample);
				float4 gpu_NN = make_float4(points[gpu_results.at(i).at(j)].x, points[gpu_results.at(i).at(j)].y, points[gpu_results.at(i).at(j)].z, gpu_results.at(i).at(j));
				float4 naive_NN = make_float4(points[naive_results.at(i).at(j)].x, points[naive_results.at(i).at(j)].y, points[naive_results.at(i).at(j)].z, naive_results.at(i).at(j));
				delta_closure.push_back(sqrt(squared_distance(point, gpu_NN)) - sqrt(squared_distance(point, naive_NN)));
			}
		}
	}
	std::cout << "#####################################################" << std::endl;
	std::cout << "#####################################################" << std::endl;
	std::cout << "#####################################################" << std::endl;
	std::cout << "Unit test completed." << std::endl;
	std::cout << "#####################################################" << std::endl;
	std::cout << "Number of NN compared: " << analyzed << std::endl;
	std::cout << "Number of NN not found by GPU but by Naive: " << gpu_not_found << std::endl;
	std::cout << "Number of NN not found by Naive but by GPU: " << naive_not_found << std::endl;
	std::cout << "Number of NN different in the two methods: " << different << std::endl;
	std::cout << "First branch boundings: " << std::endl;
	std::cout << "Left. Min: " << print_float4(tree.getNode(2).getRegion()->get_min()) << " Max: " << print_float4(tree.getNode(2).getRegion()->get_max()) << std::endl;
	std::cout << "Right. Min: " << print_float4(tree.getNode(3).getRegion()->get_min()) << " Max: " << print_float4(tree.getNode(3).getRegion()->get_max()) << std::endl;
	std::cout << "#####################################################" << std::endl;
	std::vector<int> miss_points_done;
	miss_points_done.push_back(-1);
	if (gpu_not_found > 0) {
		for (uint i = 0; i < gpu_miss_point.size(); i++) {
			if (miss_points_done.back() != (int) gpu_miss_point[i].w)
				miss_points_done.push_back((int) gpu_miss_point[i].w);
			std::cout << "GPU Missed NN for point: " << print_float4(gpu_miss_point[i]) << " for which Naive found the NN: " << print_float4(gpu_miss_naiveNN[i]) << std::endl;
		}
	}
	if (naive_not_found > 0) {
		for (uint i = 0; i < naive_miss_point.size(); i++) {
			std::cout << "NAIVE Missed NN for point: " << print_float4(naive_miss_point[i]) << " for which GPU found the NN: " << print_float4(naive_miss_gpuNN[i]) << std::endl;
		}
	}
	std::cout << "The closure distances are: " << std::endl;
	for (uint i = 0; i < delta_closure.size(); i++)
		if (delta_closure[i] != 0.0) {
			std::cout << "Broken closure!!! " << delta_closure[i] << std::endl;
			return 0;
		}
	std::cout << "All zero. Closure correct." << std::endl;
	return 0;
}

