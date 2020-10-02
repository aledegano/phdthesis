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
	string << std::setw(2) << std::setprecision(2) << print_me.w << ": " << std::setw(4) << std::setprecision(4) << print_me.x << ", " << std::setw(4) << std::setprecision(4) << print_me.y << ", " << std::setw(4) << std::setprecision(4)
			<< print_me.z << ". ";
	return string.str();
}

std::string print_float3(float4 print_me) {
	std::ostringstream string;
	string << std::setw(4) << std::setprecision(4) << print_me.x << ", " << std::setw(4) << std::setprecision(4) << print_me.y << ", " << std::setw(4) << std::setprecision(4)
			<< print_me.z << ". ";
	return string.str();
}

float evalVolume(float4 min, float4 max) {
	return (fabs(min.x - max.x) * fabs(min.y - max.y) * fabs(min.z - max.z));
}

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "This program requires exactly 1 argument: the number of points to generate." << std::endl;
		return 1;
	}
	unsigned int num_points = atoi(argv[1]);
	float4 points[num_points];
	int max_nn = num_points;

	std::random_device rd;
	std::default_random_engine generator(42); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);

	for (unsigned int i = 0; i < num_points; i++) {
		points[i].x = ceil(rnd(generator));
		points[i].y = ceil(rnd(generator));
		points[i].z = ceil(rnd(generator));
		points[i].w = i; //Use this to save the index
	}

	std::cout << "Build the tree on CPU for " << num_points << " points. And search for " << max_nn << " nearest neighbors." << std::endl;
	kdtree tree(points, num_points);
	std::cout << "Tree built with " << tree.getMaxDepth() << " layers." << std::endl;
	std::cout << std::endl;

	std::cout << "####################" << std::endl;
	std::cout << "Start health checks." << std::endl;
	std::cout << "####################" << std::endl;
	std::cout << "Leaves list:" << std::endl;
	Node cand_leaf;
	for(int i = 0; i < tree.getMaxNodes(); i++){
		cand_leaf = tree.getNode(i);
		if (cand_leaf.getLeftSon() == 0 and cand_leaf.getRightSon() == 0){
			std::cout << "Leaf " << i <<" point: " << print_float4(points[cand_leaf.getPoint()]) << " depth: " << cand_leaf.getDepth() << std::endl;
			std::cout << "Leaf region. Min: " << print_float3(cand_leaf.getRegion()->get_min()) << " Max: " << print_float3(cand_leaf.getRegion()->get_max()) << std::endl;
			std::cout << "Leaf contains point? " << cand_leaf.getRegion()->contains(points[cand_leaf.getPoint()], make_float4(0.0,0.0,0.0,0.0)) << std::endl;
		}
	}
	std::cout << "####################" << std::endl;
	std::cout << "####################" << std::endl;
	float root_volume = evalVolume(tree.getNode(0).getRegion()->get_min(), tree.getNode(0).getRegion()->get_max());
	float leaves_volume = 0;
	int leaf = 0;
	for(unsigned i = 0; i < num_points; i++){
		leaf = tree.findLeaf(i);
		std::cout << "Searched point: " << print_float4(points[i]) << std::endl;
		std::cout << "Leaf point: " << print_float4(points[tree.getNode(leaf).getPoint()]) << std::endl;
		std::cout << "Leaf region. Min: " << print_float3(tree.getNode(leaf).getRegion()->get_min()) << " Max: " << print_float3(tree.getNode(leaf).getRegion()->get_max()) << std::endl;
		std::cout << std::endl;
		leaves_volume += evalVolume(tree.getNode(leaf).getRegion()->get_min(), tree.getNode(leaf).getRegion()->get_max());
	}
	std::cout << "Volumes. Root: " << std::setw(4) << std::setprecision(4) << root_volume << " sum of leaves from Build: " << std::setw(4) << std::setprecision(4) << tree.getLeavesVolume() << " from leaves search: " << std::setw(4)
			<< std::setprecision(4) << leaves_volume << std::endl;
	std::cout << "####################" << std::endl;
}
