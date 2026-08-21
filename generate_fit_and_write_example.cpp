#include <iostream>
#include <memory>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TRandom3.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TFitResult.h>
#include <TF1.h>
#include <TPaveText.h>
#include <TString.h>

// clang++ -O3 generate_fit_and_write_example.cpp -o generate_fit_and_write_example `root-config --cflags --libs`

void generate_fit_and_save() 
{
    // 1. Configuration
    TString file_name = "gaussian_data.root";
    TString output_plot_name = "gaussian_fit_results";
    int num_events = 10000;
    double mean_gen = 5.0;      // Generated mean
    double sigma_gen = 1.5;     // Generated standard deviation

    // 2. Create ROOT File
    std::unique_ptr<TFile> output_file = std::make_unique<TFile>(file_name, "RECREATE");
    std::cout << "Created file: " << file_name << std::endl;

    // 3. Create TTree and a branch to hold the data
    std::unique_ptr<TTree> tree = std::make_unique<TTree>("tree", "Tree containing generated Gaussian data");
    tree->SetDirectory(nullptr);
    double val_buffer = 0.0;
    tree->Branch("gauss_val", &val_buffer, "gauss_val/D");

    // 4. Create Histogram
    std::unique_ptr<TH1F> hist = std::make_unique<TH1F>("h_gauss", "Gaussian Distribution & Fit;Value;Events", 100, 0, 10);
    hist->SetDirectory(nullptr);

    // 5. Generate Data, Fill Histogram, and Fill Tree
    TRandom3 rand(42); // Seed 42 for reproducibility
    std::cout << "Generating " << num_events << " events..." << std::endl;
    for (int i = 0; i < num_events; ++i) {
        val_buffer = rand.Gaus(mean_gen, sigma_gen);
        hist->Fill(val_buffer);
        tree->Fill();
    }

    // 6. Fit the Histogram using ROOT's Fit Engine
    std::cout << "\n--- Fitting Histogram ---" << std::endl;
    TFitResultPtr fit_result = hist->Fit("gaus", "S");

    // Extract Fit Results
    double chi2 = fit_result->Chi2();
    int ndof = fit_result->Ndf();
    double chi2_ndof = (ndof > 0) ? chi2 / ndof : 0;
    double fit_mean = fit_result->Parameter(1);
    double fit_mean_err = fit_result->ParError(1);
    double fit_sigma = fit_result->Parameter(2);
    double fit_sigma_err = fit_result->ParError(2);

    std::cout << "\n--- Fit Results (Terminal) ---" << std::endl;
    std::cout << "Chi2 / NDF   = " << chi2 << " / " << ndof << " = " << chi2_ndof << std::endl;
    std::cout << "Fitted Mean  = " << fit_mean << " +/- " << fit_mean_err << std::endl;
    std::cout << "Fitted Sigma = " << fit_sigma << " +/- " << fit_sigma_err << std::endl;

    // 7. Create Canvas and Plot
    gStyle->SetOptStat(1110); // Show Entries, Mean, RMS
    gStyle->SetOptFit(1111);  // Show Fit probability, Chi2/ndf, errors

    std::unique_ptr<TCanvas> canvas = std::make_unique<TCanvas>("canvas", "Gaussian Fit Canvas", 800, 600);

    // Styling the histogram
    hist->SetMarkerStyle(20);
    hist->SetMarkerSize(0.8);
    hist->SetLineColor(kBlack);

    // Style the fit line (the fit function is automatically stored in hist)
    TF1* fit_func = hist->GetFunction("gaus");
    if (fit_func) {
        fit_func->SetLineColor(kRed + 1);
        fit_func->SetLineWidth(3);
    }

    // Draw with error bars (E) and standard markers (P)
    hist->Draw("EP");

    // Add custom TPaveText to show results on the canvas
    std::unique_ptr<TPaveText> pave = std::make_unique<TPaveText>(0.15, 0.70, 0.40, 0.85, "NDC");
    pave->SetBorderSize(1);
    pave->SetLineColor(kGray);
    pave->SetFillColor(0);
    pave->SetTextAlign(12); // Left-aligned
    pave->AddText(Form("#chi^{2} / ndof = %.2f / %d", chi2, ndof));
    pave->AddText(Form("Mean = %.3f #pm %.3f", fit_mean, fit_mean_err));
    pave->AddText(Form("Sigma = %.3f #pm %.3f", fit_sigma, fit_sigma_err));
    pave->Draw();

    canvas->Update();

    // 8. Save the Plots
    canvas->SaveAs(output_plot_name + ".pdf");
    std::cout << "Plot saved as " << output_plot_name << std::endl;

    // 9. Save Tree and Histogram into the ROOT file
    tree->Write();
    hist->Write();

    // Close the file
    output_file->Close();
    std::cout << "Successfully saved tree and histogram to '" << file_name << "'" << std::endl;
}

int main()
{
    generate_fit_and_save();
    return 0;
}