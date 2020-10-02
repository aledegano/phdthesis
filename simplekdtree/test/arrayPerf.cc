#include "CudaKDtree/KDTree/src/FKDTree.h"
#include <iostream>
#include <array>
#include <utility>
#include <chrono>
#include <vector>
#include <bits/random.h>
#include <CudaKDtree/KDTree/src/CudaFKDTree.cuh>

void fkd_nth(std::vector<FKDPoint<float, 3> > input) {
	int d = 1;
	std::nth_element(input.begin(), input.end(), input.begin() + ceil(input.size() / 2), [d](const FKDPoint<float, 3>& a, const FKDPoint<float, 3>& b) -> bool {
		if(a[d] == b[d])
		return a.getId() < b.getId();
		else
		return a[d] < b[d];
	});
}

void cuda_nth(std::vector<CudaFKDPoint> input) {
	int d = 1;
	std::nth_element(input.begin(), input.end(), input.begin() + ceil(input.size() / 2), [d](const CudaFKDPoint& a, const CudaFKDPoint& b) -> bool {
		if(a[d] == b[d])
		return a.getId() < b.getId();
		else
		return a[d] < b[d];
	});
}

int main() {
	std::default_random_engine generator(42); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);
	typedef CudaFKDPoint CudaPoint;
	typedef FKDPoint<float, 3> Point;
	std::vector<CudaPoint> cuda_points;
	std::vector<Point> points;
	int repeat = 10000000;
	cuda_points.resize(repeat);
	points.resize(repeat);
	std::chrono::time_point<std::chrono::system_clock> start, end;
	float fill = 42;
	CudaPoint cuda_point(fill, fill, fill, fill);
	start = std::chrono::system_clock::now();
	for (int i = 0; i < repeat; ++i)
		cuda_points[i] = cuda_point;
	end = std::chrono::system_clock::now();
	std::cout << "[CudaFKDPoint] Initialization completed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;

	Point point(fill, fill, fill, fill);
	start = std::chrono::system_clock::now();
	for (int i = 0; i < repeat; ++i)
		points[i] = point;
	end = std::chrono::system_clock::now();
	std::cout << "[FKDPoint] Initialization completed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;

	std::cout << "##########################################" << std::endl << "Evaluating access times." << std::endl << "##########################################" << std::endl;
	std::vector<CudaFKDPoint> tmp;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < repeat; ++i)
		tmp.push_back(CudaPoint(cuda_points[i]));

	end = std::chrono::system_clock::now();
	std::cout << "[CudaFKDPoint] Reading all elements completed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;
	std::vector<Point> tmp2;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < repeat; ++i)
		tmp2.push_back(Point(points[i]));
	end = std::chrono::system_clock::now();
	std::cout << "[FKDPoint] Reading all elements completed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;
	std::cout << "##########################################" << std::endl << "Evaluating nth elements." << std::endl << "##########################################" << std::endl;
	int testNth = 10;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < testNth; ++i)
		cuda_nth(cuda_points);
	end = std::chrono::system_clock::now();
	std::cout << "[CudaFKDPoint] Evaluating nth element " << testNth << " times completed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < testNth; ++i)
		fkd_nth(points);
	end = std::chrono::system_clock::now();
	std::cout << "[FKDPoint] Evaluating nth element " << testNth << " times completed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;
	return 0;
}
