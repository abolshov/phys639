#include <iostream>
#include <memory>

#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TString.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TF1.h"
#include "TStyle.h"
#include "TFitResult.h"
#include "TLine.h"
#include "TRandom3.h"

// clang++ -O3 -march=native -flto=auto -Wall -Wextra -Wpedantic -std=c++23 hypothesis_testing.cpp -o hypothesis_testing `root-config --cflags --glibs ` -lSpectrum

double NLL(std::unique_ptr<TH1D>& hist, double prediction)
{
    int n_bins = hist->GetNbinsX();
    double res = 0.0;
    for (int i = 1; i <= n_bins; ++i)
    {
        double bin_count = hist->GetBinContent(i);
        double log_fact = 0.0;
        for (double s = 1.0; s <= bin_count; s += 1.0)
        {
            log_fact += std::log(s);
        }
        double tmp = bin_count*std::log(prediction) - prediction - log_fact;
        res += -2.0*tmp;
    }
    return res;
}

std::unique_ptr<TH1D> DoPseudoExperiments(
    std::unique_ptr<TH1D>& h, 
    int n_exp, 
    int n_bins, 
    double min, 
    double max, 
    double prediction, 
    TRandom3& rg
)
{
    auto hist = std::make_unique<TH1D>("hist", "Likelihood distribution for pseudo-data", n_bins, min, max);

    int n_bins_h = h->GetNbinsX();
    double min_h = h->GetXaxis()->GetXmin();
    double max_h = h->GetXaxis()->GetXmax();
    std::string h_name(h->GetName());

    double h_likelihood = NLL(h, prediction);
    int count = 0;

    for (int exp_num = 0; exp_num < n_exp; ++exp_num)
    {
        auto pseudo_h = std::make_unique<TH1D>("pseudo_data", "pseudo_data", n_bins_h, min_h, max_h);
        for (int i = 1; i <= n_bins_h; ++i)
        {
            pseudo_h->SetBinContent(i, rg.Poisson(prediction));
        }
        double min_FCN = NLL(pseudo_h, prediction);
        if (min_FCN >= h_likelihood)
        {
            ++count;
        }
        hist->Fill(min_FCN);
    }
    
    std::cout << h_name << " p-value = " << static_cast<double>(count)/n_exp << "\n";

    return hist;
}

int main()
{
    TFile *myFile = TFile::Open("hypothesis_testing.root");
    TStyle* gStyle = new TStyle();
    gStyle->SetPalette(1);

    TRandom3 rg;
    rg.SetSeed(0);

    auto c1 = std::make_unique<TCanvas>("c1", "c1");
    c1->SetGrid();
    c1->SetTickx();
    c1->SetTicky();
    c1->SetRightMargin(0.1);
    c1->SetLeftMargin(0.1);

    std::cout << "Fitting h1\n";
    auto h1 = std::unique_ptr<TH1D>(static_cast<TH1D*>(myFile->Get("h1")));
    h1->SetLineWidth(3);

    double min_h1 = h1->GetXaxis()->GetXmin();
    double max_h1 = h1->GetXaxis()->GetXmax();

    auto f1 = std::make_unique<TF1>("f1", "[0]", min_h1, max_h1);
    f1->SetParameter(0, 2);
    auto fit_res_1 = h1->Fit(f1.get(), "SLR");
    c1->SaveAs("h1.png");

    double h1_likelihood = NLL(h1, f1->GetParameter(0));
    std::cout << "Likelihood of h1 data = " << h1_likelihood << "\n";

    int n_pseudo_exp = 1000000;
    std::cout << "Start generating pseudo data for h1\n";
    std::cout << "n_pseudo_exp = " << n_pseudo_exp << "\n";

    auto likelihood_distr_h1 = DoPseudoExperiments(h1, n_pseudo_exp, 100, 20, 80,  f1->GetParameter(0), rg);
    likelihood_distr_h1->SetLineWidth(3);
    likelihood_distr_h1->Draw();

    int bin_max = likelihood_distr_h1->GetMaximumBin();
    double max = likelihood_distr_h1->GetBinContent(bin_max);

    auto l_h1 = std::make_unique<TLine>(h1_likelihood, 0, h1_likelihood, max);
    l_h1->SetLineColor(2);
    l_h1->SetLineWidth(3);
    l_h1->Draw("AP");

    c1->SaveAs("likelihood_distr_h1.png");

    std::cout << "Fitting h2\n";
    auto h2 = std::unique_ptr<TH1D>(static_cast<TH1D*>(myFile->Get("h2")));
    h2->SetLineWidth(3);

    double min_h2 = h2->GetXaxis()->GetXmin();
    double max_h2 = h2->GetXaxis()->GetXmax();

    auto f2 = std::make_unique<TF1>("f2", "[0]", min_h2, max_h2);
    f2->SetParameter(0, 2);
    auto fit_res_2 = h2->Fit(f2.get(), "SLR");
    c1->SaveAs("h2.png");

    double h2_likelihood = NLL(h2, f2->GetParameter(0));
    std::cout << "Likelihood of h2 data = " << h2_likelihood << "\n";

    std::cout << "Start generating pseudo data for h2\n";
    std::cout << "n_pseudo_exp = " << n_pseudo_exp << "\n";

    auto likelihood_distr_h2 = DoPseudoExperiments(h2, n_pseudo_exp, 100, 20, 80,  f2->GetParameter(0), rg);
    likelihood_distr_h2->SetLineWidth(3);
    likelihood_distr_h2->Draw();

    bin_max = likelihood_distr_h2->GetMaximumBin();
    max = likelihood_distr_h2->GetBinContent(bin_max);

    auto l_h2 = std::make_unique<TLine>(h2_likelihood, 0, h2_likelihood, max);
    l_h2->SetLineColor(2);
    l_h2->SetLineWidth(3);
    l_h2->Draw("AP");

    c1->SaveAs("likelihood_distr_h2.png");

    std::cout << "Fitting h3\n";
    auto h3 = std::unique_ptr<TH1D>(static_cast<TH1D*>(myFile->Get("h3")));
    h3->SetLineWidth(3);
    auto f3 = std::make_unique<TF1>("f2", "[0]", 0.0, 10.0);
    f2->SetParameter(0, 2);
    auto fit_res_3 = h3->Fit(f3.get(), "SLR");
    c1->SaveAs("h3.png");

    double h3_likelihood = NLL(h3, f3->GetParameter(0));
    std::cout << "Likelihood of h3 data = " << h3_likelihood << "\n";

    std::cout << "Start generating pseudo data for h3\n";
    std::cout << "n_pseudo_exp = " << n_pseudo_exp << "\n";

    auto likelihood_distr_h3 = DoPseudoExperiments(h3, n_pseudo_exp, 100, 20, 80, f3->GetParameter(0), rg);
    likelihood_distr_h3->SetLineWidth(3);
    likelihood_distr_h3->Draw();

    bin_max = likelihood_distr_h3->GetMaximumBin();
    max = likelihood_distr_h3->GetBinContent(bin_max);

    auto l_h3 = std::make_unique<TLine>(h3_likelihood, 0, h3_likelihood, max);
    l_h3->SetLineColor(2);
    l_h3->SetLineWidth(3);
    l_h3->Draw("AP");

    c1->SaveAs("likelihood_distr_h3.png");

    return 0;
}