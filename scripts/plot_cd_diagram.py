import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

INPUT_PATH = "results/cd_diagram_data.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 65)
    print("Running Statistical Tests & Plotting CD Diagram")
    print("=" * 65)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Could not find {INPUT_PATH}. Please run build_cd_diagram_data.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded data with {len(df)} rows across {len(df.columns) - 2} techniques.")

    # We need fully paired data to run Friedman + Nemenyi tests.
    test_data = df.copy()
    
    # Check for missing data
    missing = test_data.isna().sum()
    if missing.sum() > 0:
        print("\nDropping rows with missing values to ensure paired samples:")
        print(missing[missing > 0])
        test_data = test_data.dropna()

    if len(test_data) == 0:
        print("\nERROR: No remaining data after removing rows with missing values.")
        print("Cannot conduct paired statistical tests.")
        sys.exit(1)

    print(f"\nProceeding with {len(test_data)} fully paired samples (domain + country).\n")

    # Extract only the technique columns for testing
    technique_df = test_data.drop(columns=["domain", "country"])

    # 1. Run standard non-parametric tests using Autorank
    # This automatically determines if the data is normal, then uses Friedman & Nemenyi
    res = autorank(technique_df, alpha=0.05, verbose=False)

    # 2. Print statistical report
    import io
    from contextlib import redirect_stdout
    
    with io.StringIO() as buf, redirect_stdout(buf):
        create_report(res)
        report = buf.getvalue()
        
    print(report)

    # Save report to text file
    report_path = os.path.join(OUTPUT_DIR, "cd_diagram_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved statistical report to: {report_path}")

    # 3. Create Critical Differences plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_stats(res, ax=ax)
    plt.title("Critical Differences Diagram (Nemenyi Test, $\\alpha=0.05$)", pad=20, fontweight="bold")
    
    plot_path = os.path.join(OUTPUT_DIR, "cd_diagram.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved CD diagram to: {plot_path}")

    # 4. Generate LaTeX table of results
    with io.StringIO() as buf, redirect_stdout(buf):
        latex_table(res)
        latex_out = buf.getvalue()
        
    latex_path = os.path.join(OUTPUT_DIR, "cd_diagram_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_out)
    print(f"Saved LaTeX table to: {latex_path}")

if __name__ == "__main__":
    main()
