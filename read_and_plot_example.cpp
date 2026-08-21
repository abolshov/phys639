#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <iostream>

// Headers for modern ROOT GenVector
#include <Math/Vector4D.h>
#include <Math/VectorUtil.h>

// Define the type for readability (ROOT provides ROOT::Math::PtEtaPhiMVector as a shortcut)
using BJetVector = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;

// clang++ -O3 read_and_plot_example.cpp -o read_and_plot_example `root-config --cflags --libs`

void analyze_bjets_genvector() {
    // 1. Configuration
    TString input_file_name = "GluGlutoRadiontoHHto2B2Vto2B2JLNu_M_450_merge.root";
    TString tree_name = "Events";

    // 2. Open Input File
    TFile* file = TFile::Open(input_file_name, "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Could not open file " << input_file_name << std::endl;
        return;
    }

    TTree* tree = static_cast<TTree*>(file->Get(tree_name));
    if (!tree) {
        std::cerr << "Error: Tree '" << tree_name << "' not found!" << std::endl;
        file->Close();
        return;
    }

    // 3. Declare branch variables
    Float_t bjet1_pt, bjet1_eta, bjet1_phi, bjet1_mass;
    Float_t bjet2_pt, bjet2_eta, bjet2_phi, bjet2_mass;

    // 4. Link variables to Tree branches
    tree->SetBranchAddress("bjet1_pt",   &bjet1_pt);
    tree->SetBranchAddress("bjet1_eta",  &bjet1_eta);
    tree->SetBranchAddress("bjet1_phi",  &bjet1_phi);
    tree->SetBranchAddress("bjet1_mass", &bjet1_mass);

    tree->SetBranchAddress("bjet2_pt",   &bjet2_pt);
    tree->SetBranchAddress("bjet2_eta",  &bjet2_eta);
    tree->SetBranchAddress("bjet2_phi",  &bjet2_phi);
    tree->SetBranchAddress("bjet2_mass", &bjet2_mass);

    // 5. Book Histograms
    TH1F* h_mjj = new TH1F("h_mjj", "Di-bjet Invariant Mass;m_{jj} [GeV];Events", 100, 0, 600);
    TH1F* h_dR  = new TH1F("h_dR",  "Angular Separation between b-jets;#DeltaR(b_{1}, b_{2});Events", 100, 0, 5);

    // 6. Loop over events
    Long64_t n_entries = tree->GetEntries();
    std::cout << "Processing " << n_entries << " events using GenVector..." << std::endl;

    for (Long64_t i = 0; i < n_entries; ++i) {
        tree->GetEntry(i);

        // Define 4-vectors using the Pt, Eta, Phi, M constructor
        BJetVector p4_b1(bjet1_pt, bjet1_eta, bjet1_phi, bjet1_mass);
        BJetVector p4_b2(bjet2_pt, bjet2_eta, bjet2_phi, bjet2_mass);

        // Calculate Invariant Mass (Adding them yields a generic LorentzVector)
        auto p4_jj = p4_b1 + p4_b2;
        double mjj = p4_jj.M(); // or p4_jj.mass()

        // Calculate Delta R using VectorUtil helper
        double dR = ROOT::Math::VectorUtil::DeltaR(p4_b1, p4_b2);

        // Fill Histograms
        h_mjj->Fill(mjj);
        h_dR->Fill(dR);
    }

    // 7. Styling and Plotting
    gStyle->SetOptStat(0);

    // Canvas 1: Invariant Mass
    TCanvas* c1 = new TCanvas("c1", "Invariant Mass", 800, 600);
    c1->SetGrid(); 
    c1->SetLeftMargin(0.12);
    h_mjj->SetLineColor(kAzure + 1);
    h_mjj->SetFillColorAlpha(kAzure + 1, 0.15);
    h_mjj->SetLineWidth(2);
    h_mjj->GetYaxis()->SetTitleOffset(1.3);
    h_mjj->Draw("HIST");
    c1->SaveAs("dibjet_invariant_mass.pdf");

    // Canvas 2: Delta R
    TCanvas* c2 = new TCanvas("c2", "Delta R", 800, 600);
    c2->SetGrid(); 
    c2->SetLeftMargin(0.12);
    h_dR->SetLineColor(kTeal - 5);
    h_dR->SetFillColorAlpha(kTeal - 5, 0.15);
    h_dR->SetLineWidth(2);
    h_dR->GetYaxis()->SetTitleOffset(1.3);
    h_dR->Draw("HIST");
    c2->SaveAs("dibjet_deltaR.pdf");

    // 8. Cleanup
    file->Close();
    delete file;
    delete c1;
    delete c2;

    std::cout << "Done! Saved dibjet_invariant_mass.pdf and dibjet_deltaR.pdf" << std::endl;
}

int main()
{
    analyze_bjets_genvector();
    return 0;
}