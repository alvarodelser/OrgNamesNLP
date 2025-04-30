import ast
import pandas as pd
import pickle
import os
import json
import numpy as np
import random
import hashlib


def load_dataset(datafile = './data/wikidata_enriched_dataset.csv', tokenfile = './results/tokenized_names.csv', decompfile = './results/decomposed_names.csv'):
    # Load datasets
    df = pd.read_csv(datafile)
    tokens = pd.read_csv(tokenfile)
    decompositions = pd.read_csv(decompfile)
    
    # Ensure unique instances in each dataset
    df = df.drop_duplicates(subset=['instance'], keep='first')
    tokens = tokens.drop_duplicates(subset=['instance'], keep='first')
    decompositions = decompositions.drop_duplicates(subset=['instance'], keep='first')
    
    # Merge tokens on instance
    df = df.merge(tokens[['instance', 'tokenized']], on='instance', how='left')
    
    # Merge decompositions on instance
    df = df.merge(decompositions[['instance', 'decomposed']], on='instance', how='left')

    # Convert lists and ensure no duplicates in final dataset
    df['class_ids'] = df['class_ids'].apply(ast.literal_eval)
    df['classes'] = df['classes'].apply(ast.literal_eval)
    
    # Final check for duplicates
    df = df.drop_duplicates(subset=['instance'], keep='first')
    
    print(f"Loaded dataset with {len(df)} unique instances")
    return df


def get_id(experiments, domain, technique, method):
    condition = (experiments['Domain'] == domain) & (experiments['Technique'] == technique) & (experiments['Method'] == method)
    number = len(experiments.loc[condition, 'Parameters'])
    id = domain[:3] + '-' + technique[0] + '-' + method.split('_')[0] + '-' + str(number)
    print(f'Generating experiment {id}')
    return id

def load_experiments(experiments_path = "./results/experiments.csv"):
    experiments = pd.read_csv(experiments_path)
    
    # Function to safely parse parameters with NumPy arrays
    def safe_parse(param_str):
        try:
            # First try a direct literal_eval
            return ast.literal_eval(param_str)
        except (ValueError, SyntaxError) as e:
            # If that fails, it might be due to NumPy arrays in the string
            try:
                # Replace numpy array representation to make it compatible with literal_eval
                modified_str = param_str.replace('array(', 'np.array(')
                
                # Execute with numpy available in namespace
                import numpy as np
                result = eval(modified_str, {'np': np})
                return result
            except Exception as e2:
                print(f"Error parsing parameter: {param_str[:200]}...")
                print(f"Error details: {e2}")
                # Return an empty dict as fallback
                return {}
    
    # Apply the safe parsing function
    experiments['Parameters'] = experiments['Parameters'].apply(safe_parse)
    return experiments

def load_embeddings(file_path):
    """
    Load embeddings from a CSV file and transform all embedding columns.
    Embedding columns are identified by names ending with '_embedding'.
    
    Args:
        file_path (str): Path to the embeddings CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with properly formatted embeddings
    """
    # Validate inputs
    if file_path is None:
        raise ValueError("file_path must be provided")
    
    # Load embeddings file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
    embeddings_df = pd.read_csv(file_path)
    # Find all embedding columns
    embedding_columns = [col for col in embeddings_df.columns if col.endswith('_embedding')]
    if not embedding_columns:
        print(f"Warning: No embedding columns found in {file_path}")
        return embeddings_df

    # Convert stringified embeddings to lists of floats
    for col in embedding_columns:
        embeddings_df[col] = embeddings_df[col].apply(
            lambda x: np.array(ast.literal_eval(x)) if pd.notna(x) and isinstance(x, str) else None
        )
    
    return embeddings_df
    


def save_trained_model(model, exp_id, params):
    model_dir = "./results/trained_models"
    os.makedirs(model_dir, exist_ok=True)
    param_str = json.dumps(params, sort_keys=True)
    
    # Use hashlib (more reliable than hash()) and add a random component for uniqueness
    hash_obj = hashlib.md5(param_str.encode())
    random_component = random.randint(1000, 9999)
    filename = f"{model_dir}/{exp_id}_{hash_obj.hexdigest()}_{random_component}.pkl"
    
    # Save the model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    #print(f"Model saved to {filename}")
    return filename

def load_trained_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)
    

# Helper to prepare labels
def prepare_labels(df, classes, structure):
    if structure == 'nested-class':
        return df[classes].values
    
    if structure == '2-class':
        return df[classes].values.ravel()
    
    if structure == '3-multiclass':
        return df[classes].values  # multilabel