//============================================================================
// Name        : RootHisto.cpp
// Author      : Alessandro Degano
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include "TH1I.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;

int main(int argc, char** argv) {

	if (argc != 3) {
		cerr << "This program requires exactly two arguments: the files to take the input from. Aborting." << endl;
		return 1;
	}

	std::ifstream inFile1(argv[1], std::ifstream::in);
	std::ifstream inFile2(argv[2], std::ifstream::in);
	if (!inFile1 or !inFile2) {
		std::cout << "could not open one of the txt files." << std::endl;
		exit(1);
	}

	std::vector<Double_t> rechits200;
	std::vector<Double_t> rechits140;
	std::string input;

	while (inFile1 >> input)
		rechits200.push_back(atoi(input.c_str()));
	sort(rechits200.begin(), rechits200.end());

	while (inFile2 >> input)
		rechits140.push_back(atoi(input.c_str()));
	sort(rechits140.begin(), rechits140.end());

	TH1I* histo140 = new TH1I("Rechits distribution 140", "Rechits distribution", 25, 150000, 400000);
	TH1I* histo200 = new TH1I("Rechits distribution 200", "Rechits distribution", 25, 150000, 400000);

	for (auto& itr : rechits200)
		histo200->Fill(itr);
	for (auto& itr : rechits140)
		histo140->Fill(itr);

	float normalization = 100.0;

	TCanvas* canv = new TCanvas("canv", "Rechits distribution", 1024, 768);
	histo200->SetFillColor(4);
	histo200->SetBarWidth(0.4);
	histo200->SetBarOffset(0.1);
	histo140->SetFillColor(2);
	histo140->SetBarWidth(0.4);
	histo140->SetBarOffset(0.5);
	histo140->GetXaxis()->SetTitle("# Rechits");
	histo140->GetYaxis()->SetTitle("# Events/100");
	histo140->Draw("b");
	histo200->Draw("b same");
	histo140->SetStats(0);
	histo200->SetStats(0);
	histo140->Scale(normalization/histo140->Integral());
	histo200->Scale(normalization/histo200->Integral());
	TLegend *legend = new TLegend(0.7, 0.65, 0.88, 0.82);
	legend->AddEntry(histo140, "Pileup 140", "f");
	legend->AddEntry(histo200, "Pileup 200", "f");
	legend->Draw();
	canv->SaveAs("rechitsHisto.png");
	canv->SaveAs("rechitsHisto.root");

	return 0;
}
