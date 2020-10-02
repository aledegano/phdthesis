#include <cuda.h>
#include <cuda_runtime_api.h>
#include <host_defines.h>
#include <vector_functions.h>
#include <math.h>
#include <utility>
#include <iostream>
#include <vector>

__inline__
__host__ __device__
bool compare_x(float4 const & a, float4 const & b) {
	return a.x < b.x;
}
__inline__
__host__ __device__
bool compare_y(float4 const & a, float4 const & b) {
	return a.y < b.y;
}
__inline__
__host__ __device__
bool compare_z(float4 const & a, float4 const & b) {
	return a.z < b.z;
}
__inline__
__host__ __device__
bool operator==(const float4& lhs, const float4& rhs) {
	return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}
__inline__
__host__ __device__
bool operator!=(const float4& lhs, const float4& rhs) {
	return !(lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}
__inline__
__host__ __device__
bool operator<(const float4& lhs, const float4& rhs) {
	return (lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z);
}
__inline__
__host__ __device__
bool operator>(const float4& lhs, const float4& rhs) {
	return (lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z);
}
__inline__
__host__ __device__
bool operator<=(const float4& lhs, const float4& rhs) {
	return (lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z);
}
__inline__
__host__ __device__
bool operator>=(const float4& lhs, const float4& rhs) {
	return (lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z);
}
__inline__
    __host__     __device__ float4 operator+(const float4& lhs, const float4& rhs) {
	return (make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w));
}
__inline__
    __host__     __device__ float4 operator-(const float4& lhs, const float4& rhs) {
	return (make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w));
}
__inline__
    __host__     __device__ float4 operator+(const float4& lhs, const float& rhs) {
	return (make_float4(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w));
}
__inline__
    __host__     __device__ float4 operator-(const float4& lhs, const float& rhs) {
	return (make_float4(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w));
}
__inline__
__host__ __device__
float squared_distance(const float4& lhs, const float4& rhs) {
	return (pow(lhs.x - rhs.x, 2) + pow(lhs.y - rhs.y, 2) + pow(lhs.z - rhs.z, 2));
}
__inline__
__host__ __device__
float squared_distance(const float3& lhs, const float3& rhs) {
	return (pow(lhs.x - rhs.x, 2) + pow(lhs.y - rhs.y, 2) + pow(lhs.z - rhs.z, 2));
}

class BoundingRegion {
private:
	float4 dimmin;
	float4 dimmax;
public:
	__host__ __device__
	BoundingRegion() {
		dimmin = make_float4(0.0, 0.0, 0.0, 0.0);
		dimmax = make_float4(0.0, 0.0, 0.0, 0.0);
	}
	__host__ __device__
	BoundingRegion(BoundingRegion* copy) {
		dimmin = copy->get_min();
		dimmax = copy->get_max();
	}
	__host__ __device__
	BoundingRegion(float min_x, float min_y, float min_z, float max_x, float max_y, float max_z) {
		dimmin.x = min_x;
		dimmin.y = min_y;
		dimmin.z = min_z;
		dimmin.w = 0.0;
		dimmax.x = max_x;
		dimmax.y = max_y;
		dimmax.z = max_z;
		dimmax.w = 0.0;
	}

	__host__ __device__
	BoundingRegion(float4 min, float4 max) {
		dimmin = min;
		dimmax = max;
	}

	__host__ __device__
	void set_min(int axis, float this_min) {
		switch (axis) {
		case 0:
			dimmin.x = this_min;
			break;
		case 1:
			dimmin.y = this_min;
			break;
		case 2:
			dimmin.z = this_min;
			break;
		}
	}
	__host__ __device__
	void set_max(int axis, float this_max) {
		switch (axis) {
		case 0:
			dimmax.x = this_max;
			break;
		case 1:
			dimmax.y = this_max;
			break;
		case 2:
			dimmax.z = this_max;
			break;
		}
	}

	__host__ __device__
	inline float get_min(const int axis) const {
		return ((float*) (&dimmin))[axis];
	}

	__host__ __device__
	inline float get_max(const int axis) const {
		return ((float*) (&dimmax))[axis];
	}

	__host__ __device__
	inline float4 get_min() const {
		return dimmin;
	}
	__host__ __device__
	inline float4 get_max() const {
		return dimmax;
	}
	__host__ __device__
	bool contains(const float4& point, const float4& range) {
		return (point - range >= dimmin && point + range <= dimmax); //if the range touch the border we want to go up.
	}
	__host__ __device__
	bool contains(const float4& point, const float4& range_min, const float4& range_max) {
		return (point - range_min >= dimmin && point + range_max <= dimmax); //if the range touch the border we want to go up.
	}
	__host__ __device__
	bool contains(const float4& point, const float4& range, BoundingRegion* root) {
		if (root->contains(point, range))
			return (point - range >= dimmin && point + range <= dimmax); //if the range touch the border we want to go up.
		else {
			float4 new_range_min = range;
			float4 new_range_max = range;
			if (point.x - range.x < root->get_min().x)
				new_range_min.x = fabs(point.x - root->get_min().x);
			if (point.y - range.y < root->get_min().y)
				new_range_min.y = fabs(point.y - root->get_min().y);
			if (point.z - range.z < root->get_min().z)
				new_range_min.z = fabs(point.z - root->get_min().z);
			if (point.x + range.x > root->get_max().x)
				new_range_max.x = fabs(point.x - root->get_max().x);
			if (point.y + range.y > root->get_max().y)
				new_range_max.y = fabs(point.y - root->get_max().y);
			if (point.z + range.z > root->get_max().z)
				new_range_max.z = fabs(point.z - root->get_max().z);
			return this->contains(point, new_range_min, new_range_max);
		}
	}
	__host__ __device__
	bool contains(BoundingRegion* regionContained) {
		return (dimmin <= regionContained->get_min() && dimmax >= regionContained->get_max());
	}
	__host__ __device__
	bool intersect(BoundingRegion* regionIntersected) const{
		return (dimmax > regionIntersected->get_min() && dimmin < regionIntersected->get_max());
	}

};

class Node {
private:
	BoundingRegion nodeRegion;
	int leftSonId;
	int rightSonId;
	unsigned myDepth;
	int pointID;
public:
	__host__ __device__
	Node() :
			nodeRegion(BoundingRegion()) {
		leftSonId = 0;
		rightSonId = 0;
		myDepth = 0;
		pointID = -1;
	}
	__host__ __device__
	Node(BoundingRegion this_region) :
			nodeRegion(this_region) {
		leftSonId = 0;
		rightSonId = 0;
		myDepth = 0;
		pointID = -1;
	}
	__host__ __device__
	void setLeftSon(const int& this_left_son) {
		leftSonId = this_left_son;
	}
	__host__ __device__
	void setRightSon(const int& this_right_son) {
		rightSonId = this_right_son;
	}
	__host__ __device__
	void setRegion(const BoundingRegion& this_region) {
		nodeRegion = this_region;
	}
	__host__ __device__
	void setPoint(const int& this_point_id) {
		pointID = this_point_id;
	}
	__inline__ __host__ __device__
	int getPoint() {
		return pointID;
	}
	__inline__ __host__ __device__
	int getLeftSon() const {
		return leftSonId;
	}
	__inline__ __host__ __device__
	int getRightSon() const {
		return rightSonId;
	}
	__inline__     __host__     __device__ BoundingRegion* getRegion() {
		return &nodeRegion;
	}
	__host__ __device__
	void setDepth(unsigned this_depth) {
		myDepth = this_depth;
	}
	__inline__ __host__ __device__
	unsigned getDepth() const {
		return myDepth;
	}
};

class kdtree {
private:
	void Build(int, int, int, unsigned);
	void VanillaBuild(int, int, int, unsigned);
	int medianSearch(int, int, int);
	int nextId();

	Node* nodes;
	Node* gpu_nodes;
	Node root;
	int points_index[];
	int max_nodes;
	int max_depth;
	float4* points;
	float4* orig_points;
	float4* gpu_points;
	float4* gpu_range;
	int* all_nn_index;
	int* gpu_all_nn_index;
	int current_id;
	int candidate_id;
	int max_nn;
	int total_points;
	const int root_id;
	//Timers
	float search_time;
	//Debug
	int points_in_leaf;
	int n_points;
	float leaves_volume;

public:
	kdtree(float4*, int);
	~kdtree();
	Node getNode(int) const;
	float AllNNKernel(float4*, int, int);
	float AllNNSeq(float4*, int);
	float AllNNNaive(float4*, int);
	float AllNNTbbPar(float4*, int);
	float copy_points_cost;
	//debug methods
	int getMaxDepth() const;
	inline int getMaxNodes() const {
		return max_nodes;
	}
	int testTreeRewind(float4, float, bool);
	// Access methods
	int getAllNN(int, int) const;
	inline float getCopyPointsTime() const {
		return copy_points_cost;
	}
	inline float getSearchTime() const {
		return search_time;
	}
	inline int getLeavesNbr() const {
		return points_in_leaf;
	}
	// Health checks
	int findLeaf(int);
	int getNodesPoint(int) const;
	inline float getLeavesVolume() const {
		return leaves_volume;
	}
	int findContainer(int, float4);
};
