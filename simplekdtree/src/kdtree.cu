#include "kdtree.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <assert.h>
#include <driver_types.h>
#include "device_launch_parameters.h"

#include <ctime>

#include "cudaError.h"

#include "tbb/parallel_sort.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include <vector>
#include <utility>
#include <iterator>

/*
 * DEBUG
 */
#include <sstream>
#include <iomanip>
float evalVolume(float4 min, float4 max) {
	return (fabs(min.x - max.x) * fabs(min.y - max.y) * fabs(min.z - max.z));
}
std::string print_float4(float4 print_me) {
	std::ostringstream string;
	string << std::setw(2) << std::setprecision(2) << print_me.w << ": " << std::setw(4) << std::setprecision(4) << print_me.x << ", " << std::setw(4) << std::setprecision(4) << print_me.y << ", " << std::setw(4) << std::setprecision(4)
			<< print_me.z << ". ";
	return string.str();
}
std::string print_region(const BoundingRegion* print_me) {
	std::ostringstream string;
	std::ostringstream min_string;
	std::ostringstream max_string;
	for (int i = 0; i < 3; i++) {
		min_string << std::setw(4) << std::setprecision(4) << print_me->get_min(i) << ", ";
		max_string << std::setw(4) << std::setprecision(4) << print_me->get_max(i) << ", ";
	}
	string << "Min: " << min_string.str() << std::endl << "Max: " << max_string.str() << std::endl;
	return string.str();
}
/*
 * END DEBUG
 */

/*
 * Given a point find the leaf it belongs to recursively descending the tree.
 */

__host__ __device__
int findNode(Node* nodes, float4 in_point, int nodeId) {
	int left_id = nodes[nodeId].getLeftSon();
	int right_id = nodes[nodeId].getRightSon();
	if (left_id == 0 && right_id == 0) {
		return nodeId;
	}
	if (in_point >= nodes[left_id].getRegion()->get_min() && in_point < nodes[left_id].getRegion()->get_max()) {
		return findNode(nodes, in_point, left_id);
	} else {
		//If exist a right node and the point is not in the left then must be there
		return findNode(nodes, in_point, right_id);
	}
}

__host__ __device__
inline void findAllNNFromTop(Node* nodes, float4* points, const int subtree_root_id, int * const neighbors,
		const int max_nn, const int point_id, unsigned* candidate_id, const BoundingRegion& searchBox) {
	if(subtree_root_id == 0)
		neighbors[max_nn * point_id] = -1; //In case this point doesn't have NN put the trailing -1 once at the first iteration
	int depth = nodes[subtree_root_id].getDepth();
	for (int whichSon = 0; whichSon < 2; whichSon++) {
		int sonId = (whichSon == 0) ? nodes[subtree_root_id].getLeftSon() : nodes[subtree_root_id].getRightSon();
		int axis = depth % 3;
		float sonMin = nodes[sonId].getRegion()->get_min(axis);
		float sonMax = nodes[sonId].getRegion()->get_max(axis);
		bool intersection = searchBox.get_max(axis) > sonMin && searchBox.get_min(axis) < sonMax;
		if (intersection && ( *candidate_id < (max_nn - 1))) {
			if (nodes[sonId].getLeftSon() == 0 && nodes[sonId].getRightSon() == 0) {
				float4 candidatePoint = points[nodes[sonId].getPoint()];
				// if (nodes[sonId].getPoint() != point_id) Enable to discard the point itself as its own NN
				if (candidatePoint <= searchBox.get_max() && candidatePoint >= searchBox.get_min()) {  // There is still space to save the candidate
					neighbors[max_nn * point_id + *candidate_id] = nodes[sonId].getPoint();
					neighbors[max_nn * point_id + *candidate_id + 1] = -1; // Put the trailing after this NN
					(*candidate_id)++;
				}
			} else
				findAllNNFromTop(nodes, points, sonId, neighbors, max_nn, point_id, candidate_id, searchBox);
		}
	}
}

__global__
void findAllNNKernel(const int root_id, float4* in_points, Node* nodes, int* results, int max_depth,
		int res_width, float4* range, int num_points) {
	int thd_id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned candidate_id = 0;
	if (thd_id < num_points)
		findAllNNFromTop(nodes, in_points, root_id, results, res_width, thd_id, &candidate_id,
				BoundingRegion(in_points[thd_id] - range[thd_id], in_points[thd_id] + range[thd_id]));
}

struct tbbWrapper{
	Node* nodes;
	float4* orig_points;
	static const int root_id = 0;
	int* all_nn_index;
	int max_nn;
	float4* range;

	void operator()( const tbb::blocked_range<int>& points_range ) const {
        for( int i=points_range.begin(); i!=points_range.end(); ++i ){
        	unsigned cand_id = 0;
        	findAllNNFromTop(nodes, orig_points, root_id, all_nn_index, max_nn, i, &cand_id, BoundingRegion(orig_points[i] - range[i], orig_points[i] + range[i]));
        }
    }
};

/*
 * Implementation of the class kdtree.
 * The c-tor initialize the variables and move the arrays on the GPU global memory noting the time spent
 */
kdtree::kdtree(float4* in_points, int numb_points): root_id(0) {
	max_depth = ceil(log2((float) numb_points));
	max_nodes = 2 * numb_points -1;
	if (max_nodes < 1)
		return;
	total_points = numb_points;
	nodes = new Node[max_nodes];
	for (int i = 0; i < max_nodes; i++)
		nodes[i] = Node();
	points = new float4[numb_points];
	orig_points = new float4[numb_points];
	max_nn = 100;
	all_nn_index = new int[numb_points * max_nn];
	for (int i = 0; i < numb_points * max_nn; i++)
		all_nn_index[i] = -1;

	std::copy(in_points, in_points + numb_points, points);
	std::copy(in_points, in_points + numb_points, orig_points);

	BoundingRegion root_region = BoundingRegion(std::min_element(points, points + numb_points, compare_x)->x, std::min_element(points, (points + numb_points), compare_y)->y, std::min_element(points, points + numb_points, compare_z)->z,
			std::max_element(points, points + numb_points, compare_x)->x, std::max_element(points, points + numb_points, compare_y)->y, std::max_element(points, points + numb_points, compare_z)->z);
	root = Node(root_region);
	nodes[root_id] = root;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	CudaSafeCall(cudaMalloc((void ** ) &gpu_points, numb_points * sizeof(float4)));
	CudaSafeCall(cudaMemcpy(gpu_points, points, numb_points * sizeof(float4), cudaMemcpyHostToDevice));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&copy_points_cost, start, stop);

	tbb::task_scheduler_init init(32);
	current_id = root_id;
	candidate_id = 0;
	points_in_leaf = 0;
	leaves_volume = 0;
	n_points = numb_points;
	gpu_nodes = NULL;
	gpu_all_nn_index = NULL;
	gpu_range = NULL;
	Build(0, numb_points, current_id, 0);
	size_t stackSizePerThread = 4096;
	cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetLimit(cudaLimitStackSize, stackSizePerThread);
}

kdtree::~kdtree() {
	CudaSafeCall(cudaFree(gpu_points));
	if (gpu_nodes != NULL)
		CudaSafeCall(cudaFree(gpu_nodes));
	if (gpu_all_nn_index != NULL)
		CudaSafeCall(cudaFree(gpu_all_nn_index));
	delete[] nodes;
	delete[] points;
	delete[] orig_points;
	delete[] all_nn_index;
}

/*
 * Build the actual tree using the parallel sorts on CPU provided by TBB.
 * Start from the root and build all the branches recursively.
 * NOTE: each recursion is sequential, only the sort inside it is parallel.
 */
void kdtree::Build(int first_index, int last_index, int this_id, unsigned depth) { // [first_index, last_index)
	nodes[this_id].setDepth(depth);
	int numb_points = last_index - first_index;

	if (numb_points == 1) { //This node is a leaf
		nodes[this_id].setPoint((int) points[first_index].w);
		leaves_volume += evalVolume(nodes[this_id].getRegion()->get_min(), nodes[this_id].getRegion()->get_max());
		return;
	}

	BoundingRegion leftRegion(nodes[this_id].getRegion());
	BoundingRegion rightRegion(nodes[this_id].getRegion());
	int axis = depth % 3;
	//In this context median_index is referred to the absolute index of points.
	int median_index = first_index + (int) (numb_points / 2) - ((numb_points / 2) < (int) (numb_points / 2)); // == first_index + floor(numb_points / 2)
	switch (axis) {
	case 0:
		tbb::parallel_sort(points + first_index, points + last_index, compare_x);
		leftRegion.set_max(axis, points[median_index].x);
		rightRegion.set_min(axis, points[median_index].x);
		break;
	case 1:
		tbb::parallel_sort(points + first_index, points + last_index, compare_y);
		leftRegion.set_max(axis, points[median_index].y);
		rightRegion.set_min(axis, points[median_index].y);
		break;
	case 2:
		tbb::parallel_sort(points + first_index, points + last_index, compare_z);
		leftRegion.set_max(axis, points[median_index].z);
		rightRegion.set_min(axis, points[median_index].z);
		break;
	}

	assert(current_id + 2 < max_nodes);
	int leftSonId = ++current_id;
	int rightSonId = ++current_id;
	nodes[leftSonId].setRegion(leftRegion);
	nodes[rightSonId].setRegion(rightRegion);
	nodes[this_id].setLeftSon(leftSonId);
	nodes[this_id].setRightSon(rightSonId);
	Build(first_index, median_index, leftSonId, depth + 1);
	Build(median_index, last_index, rightSonId, depth + 1);
	return;
}

/*
 * Class method to call the NN kernel, allocates GPU arrays and checks time needed
 */
float kdtree::AllNNKernel(float4* range, int this_max_nn, int threads_per_block) {
	if (max_nodes < 1)
		return 0.0;
	max_nn = this_max_nn;
	float4* local_ranges = new float4[n_points];
	for (unsigned i = 0; i < n_points; i++)
		local_ranges[i] = range[i];

	CudaSafeCall(cudaMalloc((void ** ) &gpu_nodes, max_nodes * sizeof(Node)));
	delete[] all_nn_index;
	all_nn_index = new int[n_points * max_nn];
	CudaSafeCall(cudaMalloc((void** ) &gpu_all_nn_index, n_points * max_nn * sizeof(int)));
	CudaSafeCall(cudaMalloc((void ** ) &gpu_range, n_points * sizeof(float4)));
	CudaSafeCall(cudaMemcpy(gpu_nodes, nodes, max_nodes * sizeof(Node), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpu_range, local_ranges, n_points * sizeof(float4), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(gpu_all_nn_index, all_nn_index, n_points * max_nn * sizeof(int), cudaMemcpyHostToDevice));
	int num_blocks = (n_points + threads_per_block - 1) / threads_per_block;

	search_time = 0.0;
	cudaEvent_t mem_time1, mem_time2;
	cudaEventCreate(&mem_time1);
	cudaEventCreate(&mem_time2);
	cudaEventRecord(mem_time1);
	findAllNNKernel<<<num_blocks, threads_per_block>>>(root_id, gpu_points, gpu_nodes, gpu_all_nn_index, max_depth, max_nn, gpu_range, n_points);
	cudaEventRecord(mem_time2);
	cudaEventSynchronize(mem_time2);
	cudaEventElapsedTime(&search_time, mem_time1, mem_time2);

	CudaSafeCall(cudaMemcpy(all_nn_index, gpu_all_nn_index, n_points * max_nn * sizeof(int), cudaMemcpyDeviceToHost));

	if (gpu_range != NULL)
		CudaSafeCall(cudaFree(gpu_range));
	delete[] local_ranges;
	return  search_time;
}

/*
 * Class method to call the NN function in a sequential style, on CPU to compare wrt the parallelized on GPU NN search
 */
float kdtree::AllNNSeq(float4* range, int this_max_nn) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	max_nn = this_max_nn;
	delete[] all_nn_index;
	all_nn_index = new int[n_points * max_nn];
	float milliseconds = 0.0;
//	tbb::parallel_for(0, n_points, 1, [&](int i)
//	{
//		unsigned cand_id = 0;
//		findAllNNFromTop(nodes, orig_points, root_id, all_nn_index, max_nn, i, &cand_id, BoundingRegion(orig_points[i] - range[i], orig_points[i] + range[i]));
//	});

	for (unsigned int i = 0; i < n_points; i++){
		unsigned cand_id = 0;
		findAllNNFromTop(nodes, orig_points, root_id, all_nn_index, max_nn, i, &cand_id, BoundingRegion(orig_points[i] - range[i], orig_points[i] + range[i]));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	return milliseconds;
}

float kdtree::AllNNTbbPar(float4* range, int this_max_nn){
	tbbWrapper wrapper;
	max_nn = this_max_nn;
	delete[] all_nn_index;
	all_nn_index = new int[n_points * max_nn];
	wrapper.all_nn_index = all_nn_index;
	wrapper.max_nn = max_nn;
	wrapper.nodes = nodes;
	wrapper.orig_points = orig_points;
	wrapper.range = range;
	float milliseconds = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	tbb::parallel_for( tbb::blocked_range<int>( 1, max_nn ), wrapper );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	return milliseconds;
}

/*
 * Class method to search all the NN of the points in Naive way, to be used to Unit test the Kdtree functions
 */
float kdtree::AllNNNaive(float4* range, int this_max_nn) {
	max_nn = this_max_nn;
	delete[] all_nn_index;
	all_nn_index = new int[n_points * max_nn];
	int candidate_id = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for (unsigned int i = 0; i < n_points; i++) {
		candidate_id = 0;
		float4 searchBoxMax = orig_points[i] + range[i];
		float4 searchBoxMin = orig_points[i] - range[i];
		all_nn_index[i * max_nn] = -1; // Always initialize the first slot to -1 in case there is no NN for this point
		for (unsigned int j = 0; j < max_nn - 1; j++)
			if (i != j)
				if (orig_points[j] >= searchBoxMin and orig_points[j] <= searchBoxMax) {
					all_nn_index[i * max_nn + candidate_id] = j;
					all_nn_index[i * max_nn + candidate_id + 1] = -1; // This is the trailing value
					candidate_id++;
				}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float total_time = 0.0;
	cudaEventElapsedTime(&total_time, start, stop);
	return total_time;
}

/*
 * Access methods for external calls
 */
int kdtree::getAllNN(int point, int index) const {
	return all_nn_index[point * max_nn + index];
}
Node kdtree::getNode(int nodeId) const {
	assert(nodeId >= 0 && nodeId <= max_nodes);
	return nodes[nodeId];
}
int kdtree::nextId() {
	return ++current_id;
}
int kdtree::getMaxDepth() const {
	return max_depth;
}

/*
 * Health checks
 */
int kdtree::findLeaf(int point) {
	return findNode(nodes, orig_points[point], root_id);
}

int kdtree::getNodesPoint(int node_id) const {
	return nodes[node_id].getPoint();
}
