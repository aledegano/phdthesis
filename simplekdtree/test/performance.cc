 #include "CudaKDtree/KDTree/src/kdtree.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <bits/random.h>
#include <vector>
#include <sstream>
#include <iomanip>
#include <chrono>
#include "tbb/tick_count.h"

#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TLegend.h"

int main() {
	size_t stackSizePerThread = 4096;
	cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetLimit(cudaLimitStackSize, stackSizePerThread);

	constexpr int threads_per_block = 32;

	float4* points;
	float4* range;
	kdtree* tree;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

//	std::chrono::time_point<std::chrono::system_clock> start, end;
	float elapsed_seconds = 0.0;
	std::vector<int> numb_points;
	for(unsigned int i=2; i<36; i++)
		numb_points.push_back(i*10000);
	float r = 2.0;

	Double_t x_axis[numb_points.size()];
	Double_t gpu_times[numb_points.size()];
	Double_t cpu_times[numb_points.size()];
	Double_t times_err[numb_points.size()];
	std::vector<Double_t> speedup;
	std::vector<Double_t> buildTime;
	for(unsigned int i=0; i< numb_points.size(); i++)
		times_err[i] = 1.0;
//	Double_t tbb_times[numb_points.size()];
	std::random_device rd;
	std::default_random_engine generator(rd()); // rd() provides a random seed
	std::uniform_real_distribution<float> rnd(1, 100);
	std::cout << "Sizeof(int): " << sizeof(int) << " sizeof(float4): " << sizeof(float4) << " sizeof(Node): " << sizeof(Node) << std::endl;
	int i = 0;
	for (std::vector<int>::iterator num_points = numb_points.begin(); num_points != numb_points.end(); num_points++) {
		unsigned int max_nn = ceil(pow(2 * r, 3) / pow(100, 3) * *num_points * 1.5);
		std::cout << "Evaluating the performance for the search of " << max_nn << " NN of " << *num_points << " points. In a range of: "<< r << std::endl;
		points = new float4[*num_points];
		range = new float4[*num_points];
		for (int j = 0; j < *num_points; j++) {
			points[j].x = rnd(generator);
			points[j].y = rnd(generator);
			points[j].z = rnd(generator);
			points[j].w = j; //Use this to save the index
			range[j] = make_float4(r, r, r, j);
		}
		x_axis[i] = *num_points;

//		start = std::chrono::system_clock::now();
		cudaEventRecord(start);
		tree = new kdtree(points, *num_points);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		buildTime.push_back(milliseconds);
		cudaEventRecord(start);
		elapsed_seconds = tree->AllNNSeq(range, max_nn);
//		end = std::chrono::system_clock::now();
//		cpu_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cpu_times[i] = milliseconds;
		std::cout << "#############################################################" << std::endl;
		std::cout << "CPU find done for " << max_nn << " NN, took: " << elapsed_seconds << " ms." << " external: " << cpu_times[i] << " ms." << std::endl;
		std::cout << "#############################################################" << std::endl;
//		start = std::chrono::system_clock::now();
		cudaEventRecord(start);
		elapsed_seconds = tree->AllNNKernel(range, max_nn, threads_per_block);
//		end = std::chrono::system_clock::now();
//		gpu_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpu_times[i] = milliseconds;
		std::cout << "GPU find done for " << max_nn << " NN, took: " << elapsed_seconds << " ms." << " external: " << gpu_times[i] << " ms." << std::endl;
		std::cout << "#############################################################" << std::endl;
		speedup.push_back(cpu_times[i] / gpu_times[i]);

		i++;
		delete tree;
		delete[] points;
		delete[] range;
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

	graphCompare[0] = new TGraphErrors(numb_points.size(), x_axis, cpu_times, 0, times_err);
	graphCompare[0]->SetName("cpuGr");
	graphCompare[0]->SetMarkerStyle(23);
	graphCompare[0]->SetMarkerColor(4);
	graphCompare[0]->SetMarkerSize(1.8);
	graphCompare[0]->SetTitle("CPU sequential NN search");
	graphCompare[0]->SetDrawOption("AP");
	graphCompare[0]->SetFillStyle(0);
	graphCompare[0]->SetFillColor(0);
	mg->Add(graphCompare[0]);

	graphCompare[1] = new TGraphErrors(numb_points.size(), x_axis, gpu_times, 0, times_err);
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
	mg->SetMinimum(9.);
	mg->SetMaximum((ceil((cpu_times[numb_points.size() - 1] + 1000)/1000))*1000);
	mg->Draw("AP");
	mg->GetXaxis()->SetTitle("# points");
	mg->GetYaxis()->SetTitle("Time [ms]");
	mg->GetYaxis()->SetTitleOffset(0.7);
	c3->SetLogy();
	TLegend* legend = c3->BuildLegend(0.15, 0.7, 0.48, 0.9);
	legend->SetFillColor(0);
	legend->Draw();
	c3->SaveAs("volumeKdTree.root");
	c3->SaveAs("volumeKdTree.png");

	TCanvas *canv2 = new TCanvas("canv2", "canv2", 2560, 960);
	TGraph* graphSpeedup = new TGraph(numb_points.size(), x_axis, speedup.data());
	graphSpeedup->SetName("speedup");
	graphSpeedup->SetMarkerStyle(21);
	graphSpeedup->SetMarkerColor(4);
	graphSpeedup->SetTitle("Speedup");
	graphSpeedup->GetXaxis()->SetTitle("# points");
	graphSpeedup->GetYaxis()->SetTitle("Speedup");
	graphSpeedup->SetMinimum(*(std::min_element(speedup.begin(), speedup.end())) - 1.5);
	graphSpeedup->SetMaximum(*(std::max_element(speedup.begin(), speedup.end())) + 1.5);
	graphSpeedup->Draw("AP");
	canv2->SaveAs("volumeSpeedup.root");

	TCanvas *canv4 = new TCanvas("canv4", "canv4", 2000, 1500);
	TGraph* graphBuildTimes = new TGraphErrors(numb_points.size(), x_axis, buildTime.data(), 0, times_err);
	graphBuildTimes->SetName("buildTime");
	graphBuildTimes->SetMarkerStyle(21);
	graphBuildTimes->SetMarkerColor(2);
	graphBuildTimes->SetTitle("Build time");
	graphBuildTimes->GetXaxis()->SetTitle("# points");
	graphBuildTimes->GetYaxis()->SetTitle("Time [ms]");
	graphBuildTimes->SetMinimum(0.0);
	graphBuildTimes->SetMaximum(*(std::max_element(buildTime.begin(), buildTime.end())) * 1.1);
	graphBuildTimes->Draw("AP");
	canv4->SaveAs("volumeBuildTime.root");
}
