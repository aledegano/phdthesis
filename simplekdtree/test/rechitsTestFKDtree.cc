#include "CudaKDtree/KDTree/src/CudaFKDTree.cuh"
#include "CudaKDtree/KDTree/src/FKDTree.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TLegend.h"

double moliereElectrons(const unsigned int layer, double* maxRadius) {
	double constant = 2.11873;
	double slope = 0.0796206;
	double max_radius = 60.0;
	const double value = 0.1 * std::min(std::exp(constant + slope * layer), max_radius);
	// minimum radius is 1*sqrt(2+epsilon) to make sure first layer forms clusters
	double result = std::max(1.0 * std::sqrt(2.1), value);
	if(result >= max_radius)
		std::cout << "Warning max radius for layer: " << layer << " otherwise: " << std::exp(constant + slope * layer) << std::endl;
	if (result > *maxRadius)
		*maxRadius = result; //max radius for layer 24: 5.62, above stop at 6.0
	return result;
}

int main() {
	std::vector<CudaFKDPoint> gpu_points;
	std::vector<FKDPoint<float, 3> > cpu_points;
	std::vector<FKDPoint<float, 3> > cpu_search_box;
	std::vector<double> molieres;
	CudaFKDPoint* gpu_search_box;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaDeviceProp devProp;

	unsigned int* arr_cuda_results;
	float tmpLayer, tmpX, tmpY, tmpZ;
	float zBox = 1e-3;
	unsigned int maxNN = 273;
	std::vector<unsigned int> num_events;

	std::vector<Double_t> x_axis;
	std::vector<Double_t> gpu_times;
	std::vector<Double_t> cpu_times;
	std::vector<Double_t> speedup;
	std::vector<Double_t> times_err;
	std::vector<Double_t> build_times;

	//std::string inputName("/data/kdtree/extractHits/CMSSW_8_1_0_pre3/src/RecoLocalCalo/HGCalRecProducers/test/input.txt");
	std::string inputName("/afs/cern.ch/user/d/degano/input.txt");

	std::ifstream inFile(inputName.c_str(), std::ifstream::in);
	char line[1024];
	if (!inFile) {
		std::cout << "could not open txt file " << inputName << std::endl;
		exit(1);
	}
	unsigned idx;
	double maxRadius = 0;

	while (inFile.good()) {
		idx = 0;

		while (true) {
			inFile.getline(line, 1023);
			std::string lineStr(line);
			if (lineStr.empty())
				break;
			std::stringstream ss;
			ss.str(lineStr);
			std::string title;
			ss >> title;
			if (title.compare("EndEvent") == 0)
				break;
			ss >> tmpLayer >> tmpX >> tmpY >> tmpZ;
			cpu_points.push_back(FKDPoint<float, 3>(tmpX, tmpY, tmpZ, idx));
			gpu_points.push_back(CudaFKDPoint(tmpX, tmpY, tmpZ, idx));
			molieres.push_back(moliereElectrons(tmpLayer, &maxRadius));
			idx++;
		}

		if(inFile.eof())
			break;
		std::cout << "Max Moliere Radius recorded: " << maxRadius << std::endl;

		unsigned num_points = idx;
		num_events.push_back(num_points);
		CudaSafeCall(cudaMallocHost((void** )&gpu_search_box, num_points * 2 * sizeof(CudaFKDPoint)));
		cpu_search_box.resize(2 * num_points);
		for (unsigned int i = 0; i < num_points; i++) {
			times_err.push_back(1);
			gpu_search_box[2 * i] = CudaFKDPoint(gpu_points[i][0] - molieres[i], gpu_points[i][1] - molieres[i], gpu_points[i][2] - zBox, i);
			gpu_search_box[2 * i + 1] = CudaFKDPoint(gpu_points[i][0] + molieres[i], gpu_points[i][1] + molieres[i], gpu_points[i][2] + zBox, i);
			cpu_search_box[2 * i] = FKDPoint<float, 3>(gpu_search_box[2 * i][0], gpu_search_box[2 * i][1], gpu_search_box[2 * i][2], i);
			cpu_search_box[2 * i + 1] = FKDPoint<float, 3>(gpu_search_box[2 * i + 1][0], gpu_search_box[2 * i + 1][1], gpu_search_box[2 * i + 1][2], i);
		}

		CudaSafeCall(cudaMallocHost((void** )&arr_cuda_results, num_points * maxNN * sizeof(unsigned int)));

		std::cout << "Evaluating performance for " << num_points << " points in range " << *(std::max_element(molieres.begin(), molieres.end())) << " and Max Results for GPU " << maxNN << "." << std::endl;

		cudaGetDeviceProperties(&devProp, 0);
		std::cout << "Running on cuda device: " << devProp.name << std::endl;

		CudaFKDTree cudatree(num_points, gpu_points);
		cudaEventRecord(start);
		cudatree.build();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		build_times.push_back(milliseconds);

		cudaEventRecord(start);
		cudatree.search_in_the_box_linear(gpu_search_box, maxNN, arr_cuda_results);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpu_times.push_back(milliseconds);
		std::cout << "[CudaFKDTree] search completed for " << num_points << " points in: " << gpu_times.back() << " ms." << std::endl;

		FKDTree<float, 3> tree(num_points, cpu_points);
		tree.build();
		cudaEventRecord(start);
		for (unsigned int i = 0; i < num_points; ++i)
			tree.search_in_the_box_linear(cpu_search_box[2 * i], cpu_search_box[2 * i + 1]);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cpu_times.push_back(milliseconds);
		std::cout << "[FKDTree] search completed for " << num_points << " points in: " << cpu_times.back() << " ms." << std::endl;

		std::vector<unsigned int> numb_NN;
		numb_NN.resize(num_points);
		for (unsigned i = 1; i < num_points; i++) {
			numb_NN[i] = arr_cuda_results[maxNN * i];
		}
		auto maxNN = std::max_element(numb_NN.begin(), numb_NN.end());
		std::cout << "This event had a max NN count of " << *maxNN << " where the Moliere radius was: " << molieres[maxNN - numb_NN.begin()] << std::endl;

		speedup.push_back(cpu_times.back() / gpu_times.back());
		x_axis.push_back(num_points);

		CudaSafeCall(cudaFreeHost(arr_cuda_results));
		CudaSafeCall(cudaFreeHost(gpu_search_box));
		molieres.clear();
		cpu_search_box.clear();
		cpu_points.clear();
	}
	inFile.close();

	std::cout << "Print tabular for latex: " << std::endl;
	std::ostringstream str_numPoints;
	std::ostringstream str_cpu;
	std::ostringstream str_gpu;
	str_numPoints << "Points ($10^{3}$)";
	str_cpu << "CPU ($\\unit{ms}$)";
	str_gpu << "GPU ($\\unit{ms}$)";
	for(unsigned int i=0; i<num_events.size(); i+=10){
		str_numPoints << " & " << num_events[i]/1000;
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

	graphCompare[0] = new TGraphErrors(num_events.size(), x_axis.data(), cpu_times.data(), 0, times_err.data());
	graphCompare[0]->SetName("cpuGr");
	graphCompare[0]->SetMarkerStyle(23);
	graphCompare[0]->SetMarkerColor(4);
	graphCompare[0]->SetMarkerSize(1.8);
	graphCompare[0]->SetTitle("CPU sequential NN search");
	graphCompare[0]->SetDrawOption("AP");
	graphCompare[0]->SetFillStyle(0);
	graphCompare[0]->SetFillColor(0);
	mg->Add(graphCompare[0]);

	graphCompare[1] = new TGraphErrors(num_events.size(), x_axis.data(), gpu_times.data(), 0, times_err.data());
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
	mg->SetTitle("Search all neighbors");
	mg->SetMinimum(0.1);
	mg->SetMaximum((ceil((cpu_times[num_events.size() - 1] + 1000)/1000))*1000);
	mg->Draw("AP");
	mg->GetXaxis()->SetTitle("# points");
	mg->GetYaxis()->SetTitle("Time [ms]");
	mg->GetYaxis()->SetTitleOffset(0.7);
	c3->SetLogy();
	TLegend* legend = c3->BuildLegend(0.5, 0.15, 0.85, 0.4);
	legend->SetFillColor(0);
	legend->Draw();
	c3->SaveAs("rechitsplots/SearchTimes.root");
	c3->SaveAs("rechitsplots/SearchTimes.png");

	TCanvas *canv2 = new TCanvas("canv2", "canv2", 2560, 960);
	TGraph* graphSpeedup = new TGraph(num_events.size(), x_axis.data(), speedup.data());
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
	canv2->SaveAs("rechitsplots/Speedup.root");
	canv2->SaveAs("rechitsplots/Speedup.png");

	TCanvas *canv4 = new TCanvas("canv4", "canv4", 2000, 1500);
	TGraph* graphBuildTimes = new TGraphErrors(num_events.size(), x_axis.data(), build_times.data(), 0, times_err.data());
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
	canv4->SaveAs("rechitsplots/BuildTimes.root");
	canv4->SaveAs("rechitsplots/BuildTimes.png");

	return 0;
}
