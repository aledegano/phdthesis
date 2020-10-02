#include "CudaKDtree/KDTree/src/CudaFKDTree.cuh"
#include "CudaKDtree/KDTree/src/FKDTree.h"

#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <bits/random.h>
#include <vector>
#include <chrono>
#include "tbb/tick_count.h"

#include "TGraph.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TLegend.h"

int main() {
	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);

	float range = 2.0;
	std::vector<unsigned int> numb_points;
//	for(unsigned int i=3; i<26; i++)
//		numb_points.push_back(i*20000);
	numb_points.push_back(300000);

	cudaDeviceProp devProp;
	std::vector<CudaFKDPoint> cuda_points;
	CudaFKDPoint* cuda_search_box;
	unsigned int* results;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	constexpr const unsigned int numPlots = 5;
	unsigned int streams[numPlots] = {1, 2, 4, 8, 16};
	std::vector<Double_t> x_axis;
	std::vector<Double_t> times[numPlots];
	TGraph* graphCompare[numPlots];
	TMultiGraph * mg = new TMultiGraph("mg", "mg");
	int idx = 0;

	for (std::vector<unsigned int>::iterator itNumPoints = numb_points.begin(); itNumPoints != numb_points.end(); itNumPoints++) {
		unsigned int num_points = *itNumPoints;
		unsigned int maxNN = ceil(pow(2 * range, 3) / pow(100, 3) * num_points * 2.0);
		std::cout << "Evaluating performance for " << num_points << " points in range " << range << " and Max Results for GPU " << maxNN << "." << std::endl;
		if (idx > 0) {
			cuda_points.clear();
		}
		cuda_points.resize(num_points);
		CudaSafeCall(cudaMallocHost((void** )&cuda_search_box, num_points * 2 * sizeof(CudaFKDPoint)));
		for (unsigned i = 0; i < num_points; i++) {
			float x = rnd(generator);
			float y = rnd(generator);
			float z = rnd(generator);
			cuda_points[i] = CudaFKDPoint(x, y, z, i);
			for (unsigned j = 0; j < 3; ++j) {
				cuda_search_box[2 * i][j] = cuda_points[i][j] - range;
				cuda_search_box[2 * i + 1][j] = cuda_points[i][j] + range;
			}
		}

		x_axis.push_back(num_points);
		cudaGetDeviceProperties(&devProp, 0);
		std::cout << "Running on cuda device: " << devProp.name << std::endl;
		CudaFKDTree cudaTree(num_points, cuda_points);
		cudaTree.build();
		CudaSafeCall(cudaMallocHost((void** )&results, num_points * maxNN * sizeof(unsigned int)));
		float baseline = 0.0;

		for (unsigned j = 0; j < numPlots; j++) {
			cudaEventRecord(start);
			cudaTree.search_in_the_box_linear(cuda_search_box, maxNN, results, streams[j], 32);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			if(j != 0){
				times[j].push_back((baseline - milliseconds)/baseline * 100);
				std::cout << "[CudaFKDTree] search completed for " << num_points << " points in: " << times[j].back() << " ms." << std::endl;
			} else
				baseline = milliseconds;
		}
		CudaSafeCall(cudaFreeHost(results));
		CudaSafeCall(cudaFreeHost(cuda_search_box));
		++idx;
	}

	std::string titles[numPlots] = {"dull", "GPU 2 Streams", "GPU 4 Streams", "GPU 8 Streams", "GPU 16 Streams"};
	for (unsigned i = 1; i < numPlots; i++) {
		graphCompare[i] = new TGraph(idx, x_axis.data(), times[i].data());
		graphCompare[i]->SetName("Graph");
		graphCompare[i]->SetTitle(titles[i].c_str());
		graphCompare[i]->SetMarkerStyle(20+i);
		graphCompare[i]->SetLineColor(i+1);
		graphCompare[i]->SetDrawOption("AP");
		graphCompare[i]->SetFillStyle(0);
		graphCompare[i]->SetFillColor(0);
		mg->Add(graphCompare[i]);
	}

	TCanvas *canv1 = new TCanvas("canv1", "canv1", 2560, 960);
	mg->SetTitle("Search all neighbors in a box of side 4.0");
	mg->SetMinimum(0.);
	mg->SetMaximum(12.0);//(ceil((times[0].back() + 10) / 10)) * 10);
	mg->Draw("ALP");
	mg->GetXaxis()->SetTitle("# points");
	mg->GetYaxis()->SetTitle("Relative increased performance %");
	mg->GetYaxis()->SetTitleOffset(1.4);
	TLegend* legend = canv1->BuildLegend(0.15, 0.65, 0.48, 0.9);
	legend->SetFillColor(0);
	legend->Draw();
	canv1->SaveAs("fkdtreeplots/fkdStreams.root");
	canv1->SaveAs("fkdtreeplots/fkdStreams.png");
}
