import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from orgpackage.aux import load_experiments

# Canonical ordering for consistent presentation
DOMAIN_ORDER = ["medical", "administrative", "education"]
METHOD_ORDER = [
    "expert", "llm_generated", "counter_algorithm", "idf_best", 
    "0_shot", "few_shot", 
    "similarity", "classifier"
]

def generate_domain_results(csv_path="results/experiments.csv", output_dir="figures"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiments (this auto-parses Parameters into dicts)
    df = load_experiments(csv_path)
    
    # Extract structure from Parameters
    df['Structure'] = df['Parameters'].apply(lambda x: x.get('structure', 'unknown') if isinstance(x, dict) else 'unknown')
    
    # Enforce canonical domain order
    df['Domain'] = pd.Categorical(df['Domain'], categories=DOMAIN_ORDER, ordered=True)
    
    # Calculate Mean F1 and Var F1 grouped by Domain and Structure
    summary_df = df.groupby(['Domain', 'Structure'], observed=False)['F1'].agg(
        Mean_F1='mean',
        Var_F1='var'
    ).reset_index()
    
    # Sort for consistent layout (respects categorical order)
    summary_df = summary_df.sort_values(by=['Domain', 'Structure'])
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[]\n"
    latex_table += "    \\centering\n"
    latex_table += "    \\begin{tabular}{llrr}\n"
    latex_table += "        \\toprule\n"
    latex_table += "        Domain & Structure & Mean F1 & Var F1\\\\\n"
    latex_table += "        \\midrule\n"
    
    for _, row in summary_df.iterrows():
        domain = str(row['Domain']).capitalize()
        struct = row['Structure']
        mean_f1 = f"{row['Mean_F1']:.4f}" if pd.notna(row['Mean_F1']) else "0.0000"
        var_f1 = f"{row['Var_F1']:.4f}" if pd.notna(row['Var_F1']) else "0.0000"
        latex_table += f"        {domain} & {struct} & {mean_f1} & {var_f1}\\\\\n"
    
    latex_table += "        \\bottomrule\n"
    latex_table += "    \\end{tabular}\n"
    
    # Add caption in the style requested
    caption = "    \\caption{The table shows the F1 scores across domains and structures. It shows differences of scores because the structure is more or less complicated, to identify the optimal configuration.}\n"
    latex_table += caption
    latex_table += "\\end{table}\n"
    
    # Save the latex table
    table_path = os.path.join(output_dir, "domain_table.txt")
    with open(table_path, "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {table_path}")
    
    # Generate Boxplot
    # Set a scientific serious plotting style
    sns.set_theme(style="ticks", context="paper", font_scale=1.5 )
    plt.figure(figsize=(5, 3))
    
    # Create the boxplot
    ax = sns.boxplot(
        x="Domain", 
        y="F1", 
        hue="Domain",
        legend=False,
        data=df, 
        palette="colorblind", 
        linewidth=1.2,
        fliersize=2
    )
    
    plt.ylabel("F1 Score", weight='bold')
    plt.xlabel("Domain", weight='bold')
    # Add grid and despine for a cleaner look
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "domain_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Boxplot to {plot_path}")

def generate_all_results(technique, csv_path="results/experiments.csv", output_dir="figures"):
    """
    Generates LaTeX tables for each domain and a heatmap for a specific technique,
    comparing experiments against a technique-specific baseline.
    """
    import numpy as np
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiments to have access to baselines across the dataset
    all_df = load_experiments(csv_path)
    
    # Filter for the specific technique
    df = all_df[all_df['Technique'] == technique].copy()
    
    if df.empty:
        print(f"No experiments found for technique: {technique}")
        return

    # Extract common fields from Parameters
    df['Structure'] = df['Parameters'].apply(lambda x: x.get('structure', 'unknown') if isinstance(x, dict) else 'unknown')
    df['Model'] = df['Parameters'].apply(lambda x: x.get('model', 'N/A') if isinstance(x, dict) else 'N/A')
    df['Preprocessing'] = df['Parameters'].apply(lambda x: x.get('preprocessing', 'None') if isinstance(x, dict) else 'None')
    df['Tokens'] = df['Parameters'].apply(lambda x: x.get('token_num', 'N/A') if isinstance(x, dict) else 'N/A')
    df['Examples'] = df['Parameters'].apply(lambda x: x.get('n_shot', 'N/A') if isinstance(x, dict) else 'N/A')
    df['Distance'] = df['Parameters'].apply(lambda x: x.get('distance', 'N/A') if isinstance(x, dict) else 'N/A')

    # Extract numerical suffix and common ID for sorting
    df['ID_Num'] = df['ID'].apply(lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
    df['Common_ID'] = df['ID'].apply(lambda x: '-'.join(x.split('-')[1:]) if '-' in x else x)

    # Apply canonical categorical ordering
    df['Domain'] = pd.Categorical(df['Domain'], categories=DOMAIN_ORDER, ordered=True)
    df['Method'] = pd.Categorical(df['Method'], categories=METHOD_ORDER, ordered=True)

    # Define baseline lookup logic
    prefix_map = {'medical': 'med', 'administrative': 'adm', 'education': 'edu'}
    baseline_lookup = {}
    for domain in DOMAIN_ORDER:
        prefix = prefix_map.get(domain, domain[:3])
        if technique == 'rules':
            baseline_id = f"{prefix}-r-counter-0"
        elif technique == 'nli':
            baseline_id = f"{prefix}-n-0-0"
        elif technique == 'embedding':
            baseline_id = f"{prefix}-e-similarity-0"
        else:
            baseline_id = None
        
        if baseline_id:
            baseline_row = all_df[all_df['ID'] == baseline_id]
            if not baseline_row.empty:
                baseline_lookup[domain] = baseline_row.iloc[0]['F1']
            else:
                baseline_lookup[domain] = 0.0
        else:
            baseline_lookup[domain] = 0.0

    # Calculate Delta F1 (Comparison vs Baseline)
    df['Delta_F1'] = df.apply(lambda row: row['F1'] - baseline_lookup.get(row['Domain'], 0.0), axis=1)

    # Generate LaTeX Table (Technique-Specific)
    if technique == 'nli':
        headers = ["ID", "Technique", "Method", "Model", "Structure", "Domain", "Accuracy", "Recall", "F1"]
        tabular_config = "lllrrllrrr"
    elif technique == 'embedding':
        headers = ["ID", "Technique", "Method", "Model", "Structure", "Examples", "Distance", "Domain", "Accuracy", "Recall", "F1"]
        tabular_config = "lllrrlllrrr"
    else: # rules or default
        headers = ["ID", "Technique", "Method", "Structure", "Preprocessing", "Tokens", "Domain", "Accuracy", "Recall", "F1"]
        tabular_config = "lllrrrllrrr"

    latex_table = "\\begin{sidewaystable}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\small\n"
    latex_table += f"\\begin{{tabular}}{{{tabular_config}}}\n"
    latex_table += "\\toprule\n"
    latex_table += " & ".join(headers) + " \\\\\n"
    latex_table += "\\midrule\n"
    
    # Sort by Domain first, then canonical Method and ID_Num
    df_sorted = df.sort_values(by=['Domain', 'Method', 'ID_Num'])
    
    # Identify best F1 experiment per domain for bolding
    # We dropna to avoid selecting an experimental placeholder as the best
    best_f1_ids = []
    for domain in DOMAIN_ORDER:
        domain_df = df_sorted[df_sorted['Domain'] == domain]
        if not domain_df.dropna(subset=['F1']).empty:
            best_idx = domain_df['F1'].idxmax()
            best_f1_ids.append(df_sorted.loc[best_idx, 'ID'])
    
    for _, row in df_sorted.iterrows():
        is_best = row['ID'] in best_f1_ids
        
        # Build raw row values
        vals = [row['ID'], row['Technique'], row['Method']]
        
        if technique == 'nli':
            vals += [row['Model'], row['Structure']]
        elif technique == 'embedding':
            vals += [row['Model'], row['Structure'], str(row['Examples']), str(row['Distance'])]
        else: # rules
            vals += [row['Structure'], row['Preprocessing'], str(row['Tokens'])]
            
        vals += [str(row['Domain']),
                 f"{row['Accuracy']:.6f}" if pd.notna(row['Accuracy']) else "0.000000",
                 f"{row['Recall']:.6f}" if pd.notna(row['Recall']) else "0.000000",
                 f"{row['F1']:.6f}" if pd.notna(row['F1']) else "0.000000"]
        
        # Apply bold if it is the best row in domain
        if is_best:
            vals = [f"\\textbf{{{v}}}" for v in vals]
            
        latex_table += " & ".join(vals) + " \\\\\n"
        
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    
    # Caption logic
    caption = f"\\caption{{The table shows experimental metrics for {technique} across all domains. "
    caption += f"It compares results against domain-specific baselines to show performance gains. "
    caption += f"The best configuration per domain in terms of F1 is highlighted in bold.}}\n"
    
    latex_table += caption
    latex_table += "\\end{sidewaystable}\n"
    
    table_filename = f"all_{technique}_table.txt"
    table_path = os.path.join(output_dir, table_filename)
    with open(table_path, "w") as f:
        f.write(latex_table)
    print(f"Saved combined LaTeX table to {table_path}")

    # Generate Heatmap
    plt.figure(figsize=(14,4))
    sns.set_theme(style="white", context="paper")
    
    common_id_order = df_sorted['Common_ID'].unique()
    pivot_df = df.pivot_table(index='Domain', columns='Common_ID', values='Delta_F1', aggfunc='last')
    pivot_df = pivot_df.reindex(index=DOMAIN_ORDER, columns=common_id_order)
    
    # Annotation matrix
    annot_df = pivot_df.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    baseline_ids = {'rules': 'r-counter-0', 'nli': 'n-0-0', 'embedding': 'e-similarity-0'}
    target_baseline = baseline_ids.get(technique)
    if target_baseline in annot_df.columns:
        annot_df[target_baseline] = "Baseline"

    ax = sns.heatmap(
        pivot_df, 
        annot=annot_df, 
        fmt="", 
        cmap="RdBu", 
        center=0,
        linewidths=0.8,
        linecolor='white',
        cbar_kws={'label': 'F1 Difference vs Baseline'}
    )

    # Outline the baselines
    import matplotlib.patches as patches
    if target_baseline in pivot_df.columns:
        col_idx = list(pivot_df.columns).index(target_baseline)
        for row_idx in range(len(pivot_df)):
            ax.add_patch(patches.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='black', lw=4, clip_on=False))
    
    plt.xlabel("Experiment Configuration (Common ID)", weight='bold', fontsize=12)
    plt.ylabel("Domain", weight='bold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"{technique}_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap with baseline highlights and labels to {heatmap_path}")

if __name__ == '__main__':
    generate_domain_results()
    for tech in ['rules', 'nli', 'embedding']:
        generate_all_results(tech)