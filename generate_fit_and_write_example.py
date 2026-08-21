import array
import ROOT

def main():
    # 1. Configuration
    file_name = "gaussian_data.root"
    output_plot_name = "gaussian_fit_results"
    num_events = 10000
    mean_gen = 5.0  # Generated mean
    sigma_gen = 1.5  # Generated standard deviation

    # 2. Create ROOT File
    output_file = ROOT.TFile(file_name, "RECREATE")
    print(f"Created file: {file_name}")

    # 3. Create TTree and a branch to hold the data
    tree = ROOT.TTree("tree", "Tree containing generated Gaussian data")
    val_buffer = array.array("d", [0.0])
    tree.Branch("gauss_val", val_buffer, "gauss_val/D")

    # 4. Create Histogram
    hist = ROOT.TH1F(
        "h_gauss", "Gaussian Distribution & Fit;Value;Events", 100, 0, 10
    )

    # 5. Generate Data, Fill Histogram, and Fill Tree
    rand = ROOT.TRandom3(42)  # Seed 42
    print(f"Generating {num_events} events...")
    for _ in range(num_events):
        val = rand.Gaus(mean_gen, sigma_gen)
        hist.Fill(val)
        val_buffer[0] = val
        tree.Fill()

    # 6. Fit the Histogram using ROOT's Fit Engine
    print("\n--- Fitting Histogram ---")
    # "S" returns fit results, "Q" makes the fit quiet in terminal if desired
    fit_result = hist.Fit("gaus", "S")

    # Extract Fit Results
    chi2 = fit_result.Chi2()
    ndof = fit_result.Ndf()
    chi2_ndof = chi2 / ndof if ndof > 0 else 0
    fit_mean = fit_result.Parameter(1)
    fit_mean_err = fit_result.ParError(1)
    fit_sigma = fit_result.Parameter(2)
    fit_sigma_err = fit_result.ParError(2)

    print("\n--- Fit Results (Terminal) ---")
    print(f"Chi2 / NDF   = {chi2:.2f} / {ndof} = {chi2_ndof:.4f}")
    print(f"Fitted Mean  = {fit_mean:.4f} +/- {fit_mean_err:.4f}")
    print(f"Fitted Sigma = {fit_sigma:.4f} +/- {fit_sigma_err:.4f}")

    # 7. Create Canvas and Plot
    # Enable ROOT's elegant style settings
    ROOT.gStyle.SetOptStat(1110)  # Show Entries, Mean, RMS
    ROOT.gStyle.SetOptFit(1111)  # Show Fit probability, Chi2/ndf, errors

    canvas = ROOT.TCanvas("canvas", "Gaussian Fit Canvas", 800, 600)

    # Styling the histogram
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.SetLineColor(ROOT.kBlack)

    # Style the fit line (the fit function is automatically stored in hist)
    fit_func = hist.GetFunction("gaus")
    if fit_func:
        fit_func.SetLineColor(ROOT.kRed + 1)
        fit_func.SetLineWidth(3)

    # Draw with error bars (E) and standard markers (P)
    hist.Draw("EP")

    # Add custom TPaveText to show cleaner results (optional decoration)
    pave = ROOT.TPaveText(0.15, 0.70, 0.40, 0.85, "NDC")
    pave.SetBorderSize(1)
    pave.SetLineColor(ROOT.kGray)
    pave.SetFillColor(0)
    pave.SetTextAlign(12)  # Left-aligned
    pave.AddText(f"#chi^{{2}} / ndof = {chi2:.2f} / {ndof}")
    pave.AddText(f"Mean = {fit_mean:.3f} #pm {fit_mean_err:.3f}")
    pave.AddText(f"Sigma = {fit_sigma:.3f} #pm {fit_sigma_err:.3f}")
    pave.Draw()

    # Update the canvas to draw everything
    canvas.Update()

    # 8. Save the Plots
    canvas.SaveAs(f"{output_plot_name}.pdf")
    print(f"Plot saved as {output_plot_name}.pdf")

    # 9. Save objects into ROOT file
    tree.Write()
    hist.Write()

    # Close the file
    output_file.Close()
    print(f"Successfully saved tree and histogram to '{file_name}'")

if __name__ == "__main__":
    main()