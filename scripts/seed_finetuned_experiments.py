import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from orgpackage.aux import load_dataset, load_experiments, get_id, load_embeddings
from orgpackage.config import DOMAIN_CLASSES_CORR, STRUCTURE_MAPPING

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'finetuned-me5'
DOMAINS = ['medical', 'administrative', 'education']
CLASSIFIERS = ['logreg', 'svm']
EXPERIMENTS_PATH = 'results/experiments.csv'
EMBEDDING_PATH = f'results/embeddings/{MODEL_NAME}_embeddings.csv'
LABEL_EMB_PATH = 'results/embeddings/label_embeddings.csv'

def make_serializable(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    return obj

def build_splits(data):
    """Reproduce the project-standard train / validation splits."""
    logger.info("Splitting dataset into train, validation, and test sets...")
    
    # Medical & Administrative (50% test, then 50% of the remainder for train/val)
    train_medgov_full, _ = train_test_split(data, test_size=0.5, random_state=42)
    train_medgov, val_medgov = train_test_split(train_medgov_full, test_size=0.5, random_state=42)
    
    # Education (20% test, then 50% of the remainder for train/val)
    train_edu_full, _ = train_test_split(data, test_size=0.2, random_state=42)
    train_edu, val_edu = train_test_split(train_edu_full, test_size=0.5, random_state=42)
    
    trains = {
        'medical': train_medgov,
        'administrative': train_medgov,
        'education': train_edu
    }
    logger.info(f"Splits complete. Medical train size: {len(train_medgov)}, Education train size: {len(train_edu)}")
    return trains

def get_prototypes(domain, n_shot, model_name, instance_embeddings, label_embeddings, trains):
    """Generate similarity prototypes based on training set samples or labels."""
    if instance_embeddings is None:
        logger.warning(f"No instance embeddings available for {domain} domain.")
        return {}
        
    embedding_column = f"{model_name}_embedding"
    train = trains[domain].merge(
        instance_embeddings[['instance', embedding_column]], 
        on='instance', how='left'
    )
    
    # Check for missing embeddings in merged train set
    missing_count = train[embedding_column].isna().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing embeddings in {domain} train set.")
        
    prototypes = {}
    classes = DOMAIN_CLASSES_CORR.get(domain, [])
    
    if n_shot == '0_shot':
        for cls in classes:
            matching_row = label_embeddings[label_embeddings['label'] == cls.replace('_', ' ')]
            if not matching_row.empty:
                val = matching_row[embedding_column].values[0]
                prototypes[cls] = val.reshape(1, -1) if val.ndim == 1 else val
            else:
                logger.warning(f"Missing label embedding for class: {cls}")
                
    elif n_shot == '1_shot':
        for cls in classes:
            try:
                samples = train[train[cls] == 1]
                if samples.empty:
                    logger.warning(f"No training samples for class {cls} in {domain}")
                    continue
                sampled_row = samples.sample(n=1, random_state=42).iloc[0]
                val = sampled_row[embedding_column]
                prototypes[cls] = val.reshape(1, -1) if val.ndim == 1 else val
            except Exception as e:
                logger.error(f"Error sampling 1-shot prototype for {cls}: {e}")
                
    elif n_shot == 'few_shot':
        for cls in classes:
            prototypes[cls] = {}
            for country in train['country'].unique():
                try:
                    samples = train[(train[cls] == 1) & (train['country'] == country)]
                    if samples.empty: continue
                    sampled_row = samples.sample(n=1, random_state=42).iloc[0]
                    val = sampled_row[embedding_column]
                    prototypes[cls][country] = val.reshape(1, -1) if val.ndim == 1 else val
                except: continue
                
    return prototypes

def main():
    logger.info("Initializing Finetuned Experiment Seeding Script")
    
    if not os.path.exists(EXPERIMENTS_PATH):
        logger.error(f"Experiments file not found: {EXPERIMENTS_PATH}")
        return
        
    df = load_experiments(EXPERIMENTS_PATH)
    logger.info(f"Loaded existing experiments tracker with {len(df)} records.")
    
    data = load_dataset()
    trains = build_splits(data)
    
    # ------------------------------------------------------------------ #
    # LOAD EMBEDDINGS                                                    #
    # ------------------------------------------------------------------ #
    instance_embeddings = None
    label_embeddings = None
    
    if os.path.exists(EMBEDDING_PATH):
        logger.info(f"Loading finetuned embeddings from {EMBEDDING_PATH}...")
        instance_embeddings = load_embeddings(EMBEDDING_PATH)
        emb_col = f"{MODEL_NAME}_embedding"
        
        # Log embedding sample for visibility
        if emb_col in instance_embeddings.columns:
            first_emb = instance_embeddings[emb_col].dropna().iloc[0]
            logger.info(f"Embedding column '{emb_col}' details:")
            logger.info(f"  - Shape of first embedding: {first_emb.shape}")
            sample_vals = first_emb.flatten()[:5].tolist()
            logger.info(f"  - Sample of first 5 dimensions: {sample_vals}")
        
        if os.path.exists(LABEL_EMB_PATH):
            logger.info(f"Loading label embeddings from {LABEL_EMB_PATH}...")
            label_embeddings = load_embeddings(LABEL_EMB_PATH)
        else:
            logger.warning(f"Label embeddings file missing: {LABEL_EMB_PATH}")
    else:
        logger.warning(f"Embedding file missing locally: {EMBEDDING_PATH}")
        logger.warning("Prototypes will be empty placeholders.")

    # ------------------------------------------------------------------ #
    # GENERATE PENDING EXPERIMENTS                                       #
    # ------------------------------------------------------------------ #
    new_rows = []
    
    def experiment_exists(domain, method, model, n_shot=None, clf=None):
        matching = df[(df['Domain'] == domain) & (df['Method'] == method)]
        for _, row in matching.iterrows():
            p = row['Parameters']
            if p.get('model') == model:
                if method == 'similarity':
                    if p.get('n_shot') == n_shot: return True
                elif method == 'classifier':
                    if p.get('classifier') == clf: return True
        return False

    logger.info("Scanning for missing finetuned-me5 experiments...")
    
    for domain in DOMAINS:
        # Case 1: Similarity (0-shot, 1-shot, few-shot)
        for n_shot in ['0_shot', '1_shot', 'few_shot']:
            if not experiment_exists(domain, 'similarity', MODEL_NAME, n_shot=n_shot):
                eid = get_id(df, domain, 'embedding', 'similarity')
                logger.info(f"Generating prototypes for {eid} (Domain: {domain}, Shot: {n_shot})...")
                
                protos = get_prototypes(domain, n_shot, MODEL_NAME, instance_embeddings, label_embeddings, trains)
                
                # Log prototype stats
                if protos:
                    proto_keys = list(protos.keys())
                    logger.info(f"  - Generated prototypes for classes: {proto_keys}")
                    first_k = proto_keys[0]
                    if n_shot == 'few_shot':
                        logger.info(f"  - Few-shot prototype count for {first_k}: {len(protos[first_k])} countries")
                
                params = {
                    'structure': STRUCTURE_MAPPING[domain][0],
                    'model': MODEL_NAME,
                    'distance': 0.1,
                    'n_shot': n_shot,
                    'prototypes': make_serializable(protos)
                }
                new_rows.append([eid, domain, 'embedding', 'similarity', params, None, None, None])
                logger.info(f"Added pending similarity seed: {eid}")
        
        # Case 2: Classifier Head
        for clf in CLASSIFIERS:
            if not experiment_exists(domain, 'classifier', MODEL_NAME, clf=clf):
                eid = get_id(df, domain, 'embedding', 'classifier')
                params = {
                    'structure': STRUCTURE_MAPPING[domain][-1],
                    'model': MODEL_NAME,
                    'classifier': clf
                }
                new_rows.append([eid, domain, 'embedding', 'classifier', params, None, None, None])
                logger.info(f"Added pending classifier seed: {eid} ({clf})")

    # ------------------------------------------------------------------ #
    # SAVE UPDATED EXPERIMENTS                                           #
    # ------------------------------------------------------------------ #
    if new_rows:
        logger.info(f"Seeding completed. Found {len(new_rows)} new experiment records locally.")
        new_df = pd.DataFrame(new_rows, columns=df.columns)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        
        logger.info(f"Saving changes to {EXPERIMENTS_PATH}...")
        # Ensure parameters are safely stringified (matching project standard)
        updated_df['Parameters'] = updated_df['Parameters'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        )
        updated_df.to_csv(EXPERIMENTS_PATH, index=False)
        logger.info("Successfully updated results/experiments.csv")
    else:
        logger.info("No new experiments found for seeding. Tracker is up to date.")

if __name__ == "__main__":
    main()
