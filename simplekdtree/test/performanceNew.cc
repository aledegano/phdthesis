#include "CudaKDtree/KDTree/src/CudaFKDTree.cuh"
#include "CudaKDtree/KDTree/src/FKDTree.h"

#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <bits/random.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "tbb/tick_count.h"

#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TLegend.h"

int main() {
	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);

	unsigned int testType = 0; // 1 to scan points number, 2 to scan ranges

	testType = 1;

	std::vector<float> ranges;
	std::vector<unsigned int> numb_points;

	std::vector<CudaFKDPoint> cuda_points;
	CudaFKDPoint* cuda_search_box;
	unsigned int* results = nullptr;
	std::vector<FKDPoint<float, 3> > points;
	std::vector<FKDPoint<float, 3> > search_box;
	std::string graphTitle;
	std::string xAxisTitle;

	if (testType == 1) {
		graphTitle = "Search all neighbors in a box of side 4.0";
		xAxisTitle = "# points";
		ranges.push_back(2.0);
		for(unsigned int i=2; i<51; i++)
			numb_points.push_back(i*10000);
	} else if (testType == 2) {
		graphTitle = "Search all neighbors of 300k points";
		xAxisTitle = "side of search cube";
		numb_points.push_back(300000);
		ranges.push_back(1.0);
		ranges.push_back(2.0);
		ranges.push_back(3.0);
		ranges.push_back(4.0);
		ranges.push_back(5.0);
	}

//	std::chrono::time_point<std::chrono::system_clock> start, end, start2, end2;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaDeviceProp devProp;

	std::vector<Double_t> x_axis;
	std::vector<Double_t> gpu_times;
	std::vector<Double_t> cpu_times;
	std::vector<Double_t> speedup;
	std::vector<Double_t> times_err;
	std::vector<Double_t> build_times;
	const unsigned int maxNN = 200;

	unsigned int idx = 0;
	for (std::vector<unsigned int>::iterator itNumPoints = numb_points.begin(); itNumPoints != numb_points.end(); itNumPoints++) {
		for (std::vector<float>::iterator itRanges = ranges.begin(); itRanges != ranges.end(); itRanges++) {
			unsigned int num_points = *itNumPoints;
			float range = *itRanges;
//			unsigned int maxNN = ceil(pow(2 * range, 3) / pow(100, 3) * num_points * 1.5);
			CudaSafeCall(cudaMallocHost((void** )&results, num_points * maxNN * sizeof(unsigned int)));
			std::cout << "Evaluating performance for " << num_points << " points in range " << range << " and Max Results for GPU " << maxNN << "." << std::endl;
			if (idx > 0) {
				points.clear();
				cuda_points.clear();
				search_box.clear();
			}
			points.resize(num_points);
			cuda_points.resize(num_points);
			search_box.resize(2 * num_points);
			CudaSafeCall(cudaMallocHost((void** )&cuda_search_box, num_points * 2 * sizeof(CudaFKDPoint)));
			for (unsigned i = 0; i < num_points; i++) {
				float x = rnd(generator);
				float y = rnd(generator);
				float z = rnd(generator);
				points[i] = FKDPoint<float, 3>(x, y, z, i);
				cuda_points[i] = CudaFKDPoint(x, y, z, i);
				for (unsigned j = 0; j < 3; ++j) {
					cuda_search_box[2 * i][j] = cuda_points[i][j] - range;
					search_box[2 * i][j] = cuda_search_box[2 * i][j];
					cuda_search_box[2 * i + 1][j] = cuda_points[i][j] + range;
					search_box[2 * i + 1][j] = cuda_search_box[2 * i + 1][j];
				}
			}

			if (testType == 1)
				x_axis.push_back(num_points);
			else if (testType == 2)
				x_axis.push_back(2 * range);

			cudaGetDeviceProperties(&devProp, 0);
			std::cout << "Running on cuda device: " << devProp.name << std::endl;

			CudaFKDTree cudaTree(num_points, cuda_points);
			cudaEventRecord(start);
			cudaTree.build();
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			build_times.push_back(milliseconds);

			cudaEventRecord(start);
			cudaTree.search_in_the_box_linear(cuda_search_box, maxNN, results);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			gpu_times.push_back(milliseconds);
			std::cout << "[CudaFKDTree] search completed for " << num_points << " points in: " << gpu_times[idx] << " ms." << std::endl;

			FKDTree<float, 3> tree(num_points, points);
			tree.build();
			cudaEventRecord(start);
			for (unsigned int i = 0; i < num_points; ++i)
				tree.search_in_the_box_linear(search_box[2 * i], search_box[2 * i + 1]);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			cpu_times.push_back(milliseconds);
			std::cout << "[FKDTree] search completed for " << num_points << " points in: " << cpu_times[idx] << " ms." << std::endl;

			speedup.push_back(cpu_times[idx] / gpu_times[idx]);
			CudaSafeCall(cudaFreeHost(cuda_search_box));
			CudaSafeCall(cudaFreeHost(results));
			++idx;
		}
	}

	std::cout << "Print tabular for latex: " << std::endl;
	std::ostringstream str_numPoints;
	std::ostringstream str_cpu;
	std::ostringstream str_gpu;
	str_numPoints << "Points ($10^{3}$)";
	str_cpu << "CPU ($\\unit{ms}$)";
	str_gpu << "GPU ($\\unit{ms}$)";
	for(unsigned int i=3; i<numb_points.size(); i+=5){
		str_numPoints << " & " << numb_points[i]/1000;
		str_cpu << " & " << std::setprecision(4) << cpu_times[i];
		str_gpu << " & " << std::setprecision(4) << gpu_times[i];
	}
	str_numPoints << " \\\\";
	str_cpu << " \\\\";
	str_gpu << " \\\\";
	std::cout << str_numPoints.str() << std::endl;
	std::cout << str_cpu.str() << std::endl;
	std::cout << str_gpu.str() << std::endl;

	TMultiGraph * mg = new TMultiGraph("mg", "mg");
	TGraph* graphCompare[2];

	graphCompare[0] = new TGraphErrors(numb_points.size(), x_axis.data(), cpu_times.data(), 0, times_err.data());
	graphCompare[0]->SetName("cpuGr");
	graphCompare[0]->SetMarkerStyle(23);
	graphCompare[0]->SetMarkerColor(4);
	graphCompare[0]->SetMarkerSize(1.8);
	graphCompare[0]->SetTitle("CPU sequential NN search");
	graphCompare[0]->SetDrawOption("AP");
	graphCompare[0]->SetFillStyle(0);
	graphCompare[0]->SetFillColor(0);
	mg->Add(graphCompare[0]);

	graphCompare[1] = new TGraphErrors(numb_points.size(), x_axis.data(), gpu_times.data(), 0, times_err.data());
	graphCompare[0]->SetName("gpuGr");
	graphCompare[1]->SetMarkerStyle(22);
	graphCompare[1]->SetMarkerColor(2);
	graphCompare[1]->SetMarkerSize(1.8);
	graphCompare[1]->SetTitle("GPU parallel NN search");
	graphCompare[1]->SetDrawOption("AP");
	graphCompare[1]->SetFillStyle(0);
	graphCompare[1]->SetFillColor(0);
	mg->Add(graphCompare[1]);

	TCanvas *c3 = new TCanvas("c3", "c3", 2560, 960);
	mg->SetTitle("Search all neighbors in a box of side 4.0");
	mg->SetMinimum(0.1);
	mg->SetMaximum((ceil((cpu_times[numb_points.size() - 1] + 1000)/1000))*1000);
	mg->Draw("AP");
	mg->GetXaxis()->SetTitle("# points");
	mg->GetYaxis()->SetTitle("Time [ms]");
	mg->GetYaxis()->SetTitleOffset(0.7);
	c3->SetLogy();
	TLegend* legend = c3->BuildLegend(0.5, 0.15, 0.85, 0.4);
	legend->SetFillColor(0);
	legend->Draw();
	c3->SaveAs("fkdtreeplots/fkdSearchTimes.root");
	c3->SaveAs("fkdtreeplots/fkdSearchTimes.png");

	TCanvas *canv2 = new TCanvas("canv2", "canv2", 2560, 960);
	TGraph* graphSpeedup = new TGraph(numb_points.size(), x_axis.data(), speedup.data());
	graphSpeedup->SetName("speedup");
	graphSpeedup->SetMarkerStyle(21);
	graphSpeedup->SetMarkerColor(4);
	graphSpeedup->SetMarkerSize(1.8);
	graphSpeedup->SetTitle("Speedup");
	graphSpeedup->GetXaxis()->SetTitle("# points");
	graphSpeedup->GetYaxis()->SetTitle("Speedup");
	graphSpeedup->SetMinimum(*(std::min_element(speedup.begin(), speedup.end())) - 1.5);
	graphSpeedup->SetMaximum(*(std::max_element(speedup.begin(), speedup.end())) + 1.5);
	graphSpeedup->Draw("AP");
	canv2->SaveAs("fkdtreeplots/fkdSpeedup.root");
	canv2->SaveAs("fkdtreeplots/fkdSpeedup.png");

	TCanvas *canv4 = new TCanvas("canv4", "canv4", 2000, 1500);
	TGraph* graphBuildTimes = new TGraphErrors(numb_points.size(), x_axis.data(), build_times.data(), 0, times_err.data());
	graphBuildTimes->SetName("buildTime");
	graphBuildTimes->SetMarkerStyle(21);
	graphBuildTimes->SetMarkerColor(2);
	graphBuildTimes->SetMarkerSize(1.8);
	graphBuildTimes->SetTitle("Build time");
	graphBuildTimes->GetXaxis()->SetTitle("# points");
	graphBuildTimes->GetYaxis()->SetTitle("Time [ms]");
	graphBuildTimes->SetMinimum(0.0);
	graphBuildTimes->SetMaximum(*(std::max_element(build_times.begin(), build_times.end())) * 1.1);
	graphBuildTimes->Draw("AP");
	canv4->SaveAs("fkdtreeplots/fkdBuildTimes.root");
	canv4->SaveAs("fkdtreeplots/fkdBuildTimes.png");
}
