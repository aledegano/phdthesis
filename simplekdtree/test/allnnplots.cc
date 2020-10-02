#include "CudaKDtree/KDTree/src/kdtree.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <bits/random.h>
#include <vector>

#include "TGraph.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TLegend.h"

int main() {
	unsigned int num_points = 100000;
	float4 points[num_points];
	float4 range[num_points];
	unsigned int max_nn = 0;
	constexpr int threads_per_block = 1024;
	int samples = 10;
	Double_t x_axis[samples];
	Double_t gpu_times[samples];
	Double_t cpu_times[samples];

	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);

	for (unsigned int i = 0; i < num_points; i++) {
		points[i].x = rnd(generator);
		points[i].y = rnd(generator);
		points[i].z = rnd(generator);
		points[i].w = i; //Use this to save the index
		range[i] = make_float4(10.1, 10.1, 10.1, i);
	}

	std::cout << "Build the tree on CPU for " << num_points << " points." << std::endl;
	kdtree tree(points, num_points);
	std::cout << "Tree built with " << tree.getMaxDepth() << " layers." << std::endl;
	std::cout << std::endl;

	size_t stackSizePerThread = 4096;
	cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetLimit(cudaLimitStackSize, stackSizePerThread);

	for(int i=0; i<samples; i++){
		max_nn = (i+1)*10;
		x_axis[i] = max_nn;
		cpu_times[i] = tree.AllNNSeq(range, max_nn);
		std::cout << "CPU find done for " << max_nn << " NN, took: " << cpu_times[i] << "ms." << std::endl;
		gpu_times[i] = tree.AllNNKernel(range, max_nn, threads_per_block);
		std::cout << "GPU find done for " << max_nn << " NN, took: " << gpu_times[i] << "ms." << std::endl;
	}

	TMultiGraph * mg = new TMultiGraph("mg", "mg");
	TGraph* graphCompare[2];

	graphCompare[0] = new TGraph(samples, x_axis, cpu_times);
	graphCompare[0]->SetMarkerStyle(21);
	graphCompare[0]->SetLineColor(2);
	graphCompare[0]->SetTitle("kdtree CPU");
	graphCompare[0]->SetDrawOption("AP");
	graphCompare[0]->SetFillStyle(0);
	graphCompare[0]->SetFillColor(0);
	mg->Add(graphCompare[0]);

	graphCompare[1] = new TGraph(samples, x_axis, gpu_times);
	graphCompare[1]->SetMarkerStyle(21);
	graphCompare[1]->SetLineColor(3);
	graphCompare[1]->SetTitle("kdtree GPU");
	graphCompare[1]->SetDrawOption("AP");
	graphCompare[1]->SetFillStyle(0);
	graphCompare[1]->SetFillColor(0);
	mg->Add(graphCompare[1]);

	TCanvas *c3 = new TCanvas("c3", "c3", 1280, 960);
	mg->SetTitle("k Nearest Neighbors search");
	mg->SetMinimum(0.);
	mg->SetMaximum(500.);
	mg->Draw("ALP");
	mg->GetXaxis()->SetTitle("# Nearest Neighbors");
	mg->GetYaxis()->SetTitle("Time [ms]");
	mg->GetYaxis()->SetTitleOffset(1.4);
	TLegend* legend = c3->BuildLegend();
	legend->SetFillColor(0);
	c3->SaveAs("allNNplot.root");
	c3->SaveAs("allNNplot.png");
	return 0;
}
