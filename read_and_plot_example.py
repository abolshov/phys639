import matplotlib.pyplot as plt
import numpy as np
import uproot

def main():
    # 1. Configuration
    file_name = "TTto4Q_merge.root"
    tree_name = "Events"
    output_pdf = "top_masses_comparison_uproot.pdf"

    # 2. Read data from ROOT file using Uproot
    print(f"Opening {file_name}...")
    with uproot.open(f"{file_name}:{tree_name}") as tree:
        # Read the branches directly into NumPy arrays
        # library="np" ensures we get standard numpy arrays
        data = tree.arrays(["hadT_mass", "lepT_mass"], library="np")

    hadT_mass = data["hadT_mass"]
    lepT_mass = data["lepT_mass"]

    print(f"Loaded {len(hadT_mass)} events.")

    # 3. Define binning (e.g., 100 bins between 0 and 1500 GeV)
    bins = np.linspace(0, 1500, 101)

    # 4. Create the Plot
    plt.figure(figsize=(8, 6), dpi=100)

    # Plot Hadronic Top (Step histogram, filled with alpha transparency)
    plt.hist(
        hadT_mass,
        bins=bins,
        histtype="step",
        color="crimson",
        linewidth=1.8,
        label="Hadronic Top",
    )

    # Plot Leptonic Top
    plt.hist(
        lepT_mass,
        bins=bins,
        histtype="step",
        color="royalblue",
        linewidth=1.8,
        label="Leptonic Top",
    )

    # 5. Styling and Labels
    plt.title("Top Quark Reconstructed Mass Comparison", fontsize=14, pad=15)
    plt.xlabel("Mass [GeV]", fontsize=12)
    plt.ylabel(f"Events / {bins[1]-bins[0]:.1f} GeV", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=11)

    # 6. Save and Show
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf")
    print(f"Plot saved successfully to {output_pdf}")

if __name__ == "__main__":
    main()