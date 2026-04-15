import os
import pandas as pd

from orgpackage.tester import (
    build_continuous_correctness_table, 
    run_all_pairwise_tests, 
    plot_permutation_heatmap
)

def main():
    print("Loading NLI Confidences Dataset...")
    csv_path = 'results/nli_confidences.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Use low_memory=False to avoid mixed dtype warnings on large CSV files
    conf = pd.read_csv(csv_path, low_memory=False)

    domains_to_run = [
        ('medical', 'med-n-'),
        ('administrative', 'adm-n-'),
        ('education', 'edu-n-')
    ]

    # Map experiment suffixes to standard model names across all domains
    # Excludes med-n-0-4 to 7
    suffix_to_model = {
        '0-0': 'roberta-large',
        '0-1': 'bge-m3',
        '0-2': 'mDeBerta',
        '0-3': 'MiniLM',
        'few-0': 'Qwen/DeepSeek (Generative)' 
    }

    strata = []

    for domain, prefix in domains_to_run:
        print(f"\n" + "="*50)
        print(f"PROCESSING DOMAIN: {domain} (Prefix: {prefix})")
        print("="*50)

        # Identify the domain subset
        nli_subset_df = conf.dropna(subset=[col for col in conf.columns if col.startswith(prefix)]).copy()

        if nli_subset_df.empty:
            print(f"Warning: No instances identified for domain {domain} with prefix {prefix}.")
            continue

        print("Building Continuous Correctness Table...")
        try:
            correctness_df = build_continuous_correctness_table(
                domain=domain,
                nli_df=nli_subset_df,
                experiment_prefix=prefix
            )
        except Exception as e:
            print(f"Error building correctness table for {domain}: {e}")
            continue

        # Filter and rename experiments to standard model names
        columns_to_keep = {}
        for col in correctness_df.columns:
            suffix = col[len(prefix):]
            if suffix in suffix_to_model:
                # Exclusion rule requested by user
                if domain == 'medical' and suffix in ['0-4', '0-5', '0-6', '0-7']:
                    continue
                columns_to_keep[col] = suffix_to_model[suffix]

        # Apply filtering and renaming
        stratum_df = correctness_df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
        
        # Add the entire domain as a single stratum
        if not stratum_df.empty:
            strata.append(stratum_df)
            print(f"Successfully processed {domain} as a single stratum.")
            # Note: if few-0 predictions are not in nli_confidences.csv, they won't appear here, protecting the code from crashing.
            print(f"Found experiments: {list(stratum_df.columns)}") 
        else:
            print(f"Warning: Stratum for {domain} is empty after filtering.")

    if len(strata) < 2:
        print("\nError: Need at least 2 valid strata (domains) for meaningful cross-domain testing.")
        return

    print("\n" + "="*50)
    print(f"Running Cross-Domain Permutations with {len(strata)} strata (domains)...")
    print("="*50)

    try:
        results_df = run_all_pairwise_tests(
            strata=strata,
            n_perm=10_000,
            random_state=42,
            statistic="pooled"
        )
    except Exception as e:
        print(f"Error running cross-domain permutation tests: {e}")
        return
    
    print(f"\nCross-Domain Pairwise Test Results:")
    print(results_df.head(20))

    # Plot Heatmap
    output_path = "figures/nli_continuous_heatmap_cross_domain.png"
    os.makedirs("figures", exist_ok=True)
    print(f"\nPlotting heatmap to {output_path}...")
    try:
        plot_permutation_heatmap(
            results=results_df,
            title="Cross-Domain NLI Model Comparison (Domain Stratified)\nNote: Generative classif. scores used instead of inference",
            save_path=output_path,
            figsize=(8, 6)
        )
        print("Heatmap generated successfully.")
    except Exception as e:
        print(f"Error plotting heatmap: {e}")
        
    print("Done!")

if __name__ == '__main__':
    main()
