import ast
import pandas as pd

def get_id(experiments, domain, technique, method):
    condition = (experiments['Domain'] == domain) & (experiments['Technique'] == technique) & (experiments['Method'] == method)
    number = len(experiments.loc[condition, 'Parameters'])
    id = domain[:3] + '-' + technique[0] + '-' + method.split('_')[0] + '-' + str(number)
    print(f'Generating experiment {id}')
    return id

def load_experiments(experiments_path = "./results/experiments.csv"):
    experiments = pd.read_csv(experiments_path)
    experiments['Parameters'] = experiments['Parameters'].apply(ast.literal_eval)
    return experiments

def load_dataset(datafile = './data/wikidata_enriched_dataset.csv', tokenfile = './results/tokenized_names.csv', decompfile = './results/decomposed_names.csv'):
    df = pd.read_csv(datafile)
    tokens =  pd.read_csv(tokenfile)
    decompositions = pd.read_csv('./results/decomposed_names.csv')
    df = df.merge(tokens[['instance', 'tokenized']], on='instance', how='left')
    df = df.merge(decompositions[['names', 'decomposed']], on='names', how='left')

    df['class_ids'] = df['class_ids'].apply(ast.literal_eval)
    df['classes'] = df['classes'].apply(ast.literal_eval)
    return df
