import os, sys
PROJECT_ROOT = os.path.expanduser("~/notebooks/SWJ")
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    
import pandas as pd
import ollama
from sklearn.model_selection import train_test_split

from orgpackage.aux import load_dataset, load_experiments, get_id
from orgpackage.config import DOMAIN_CLASSES_CORR, STRUCTURE_MAPPING
from orgpackage.trainer import build_few_shot_prompt
from orgpackage.evaluator import evaluate_ollama_experiment


data = load_dataset()

train_medgov_full, test_medgov = train_test_split(data, test_size=0.5, random_state=42)
train_medgov, val_medgov = train_test_split(train_medgov_full, test_size=0.5, random_state=42)

train_edu_full, test_edu = train_test_split(data, test_size=0.2, random_state=42)
train_edu, val_edu = train_test_split(train_edu_full, test_size=0.5, random_state=42)

trains = {
    'medical': train_medgov,
    'administrative': train_medgov,
    'education': train_edu
}

tests = {
    'medical': test_medgov,
    'administrative': test_medgov,
    'education': test_edu
}

print(f"Train sizes: medical/admin={len(train_medgov)}, education={len(train_edu)}")
print(f"Test sizes:  medical/admin={len(test_medgov)}, education={len(test_edu)}")

OLLAMA_HOST = "http://localhost:11435"
MODEL_NAME = "DeepSeek-R1:7B"

client = ollama.Client(host=OLLAMA_HOST)

try:
    models = client.list()
    available = [m.model for m in models.models]
    print(f"Available models: {available}")
    assert any(MODEL_NAME.lower() in m.lower() for m in available), (
        f"{MODEL_NAME} not found. Pull it with: ollama pull {MODEL_NAME}"
    )
    print(f"Using model: {MODEL_NAME}")
except Exception as e:
    print(f"Connection error: {e}")
    print(f"Make sure Ollama is running on {OLLAMA_HOST}")

TEST_SAMPLE_SIZE = None  # Set to e.g. 500 for a quick run, None for full test set

experiments_path = './results/experiments.csv'
experiments = load_experiments(experiments_path)

for domain in ['medical', 'administrative', 'education']:
    classes = DOMAIN_CLASSES_CORR[domain]
    structure = STRUCTURE_MAPPING[domain][-1]

    prompt = build_few_shot_prompt(trains[domain], domain, classes)

    # Force ID to xxx-n-few-0 to overwrite
    exp_id = f"{domain[:3]}-n-few-0"
    params = {
        'structure': structure,
        'model': MODEL_NAME,
        'prompt': prompt
    }

    exp = pd.Series({
        'ID': exp_id,
        'Domain': domain,
        'Technique': 'nli',
        'Method': 'few_shot',
        'Parameters': params,
        'Accuracy': None,
        'Recall': None,
        'F1': None
    })

    test_df = tests[domain].copy()
    if TEST_SAMPLE_SIZE is not None:
        test_df = test_df.sample(n=min(TEST_SAMPLE_SIZE, len(test_df)), random_state=42)

    print(f"\n{'='*60}")
    print(f"Domain: {domain.upper()} — {len(test_df)} instances, classes: {classes}")
    print(f"Experiment ID: {exp_id}")
    print(f"{'='*60}")

    exp = evaluate_ollama_experiment(exp, test_df, client, MODEL_NAME)

    # Save per-experiment predictions (including pseudo-confidence columns)
    pred_cols = [c for c in test_df.columns if c.startswith(exp_id)]
    out_cols = ['instance', 'country'] + pred_cols
    out_path = f'./results/ollama_{exp_id}_predictions.csv'
    test_df[out_cols].to_csv(out_path, index=False)
    print(f"  Predictions saved to {out_path}")

    # Overwrite if ID exists, otherwise append
    condition = experiments['ID'] == exp_id
    if condition.any():
        # Update existing row
        idx = experiments[condition].index[0]
        experiments.loc[idx] = exp
    else:
        new_row = pd.DataFrame([exp])
        experiments = pd.concat([experiments, new_row], ignore_index=True)
        
    experiments.to_csv(experiments_path, index=False)

    print(f"  Accuracy: {exp['Accuracy']:.4f}")
    print(f"  Recall:   {exp['Recall']:.4f}")
    print(f"  F1:       {exp['F1']:.4f}")
    print(f"  Saved to {experiments_path}")

from orgpackage.config import COUNTRY_DICT

experiments = load_experiments(experiments_path)
few_shot_exps = experiments[experiments['Method'] == 'few_shot']
display(few_shot_exps[['ID', 'Domain', 'Technique', 'Method', 'Accuracy', 'Recall', 'F1']])

for _, exp in few_shot_exps.iterrows():
    print(f"\n{'='*60}")
    print(f"{exp['ID']} — {exp['Domain'].upper()}")
    print(f"{'='*60}")
    ca = exp['Parameters'].get('country_accuracy', {})
    cf = exp['Parameters'].get('country_f1', {})
    rows = []
    for cid in sorted(ca.keys()):
        rows.append({
            'Country': COUNTRY_DICT.get(cid, {}).get('country', cid),
            'Accuracy': round(ca[cid], 4) if ca[cid] is not None else '-',
            'F1': round(cf.get(cid), 4) if cf.get(cid) is not None else '-'
        })
    display(pd.DataFrame(rows))
