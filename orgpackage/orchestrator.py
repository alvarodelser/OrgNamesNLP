import json
import os
import ast
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from orgpackage.aux import load_experiments, get_id, save_trained_model, load_embeddings
from orgpackage.config import STRUCTURE_MAPPING, NLI_MODELS, DOMAIN_CLASSES_CORR, EMB_MODELS
from orgpackage.evaluator import evaluate_classifier_experiment
from orgpackage.trainer import train_rules, optimize_parameter, train_classifier




############################################################# RULES ######################################################
def rule_loader():
    def get_keywords(domain, method):
        parameters = {}
        for category in DOMAIN_CLASSES_CORR.get(domain, []):
            file_name = f'./keywords/{method}/{category}_whitelist.json'
            with open(file_name, 'r') as f:
                parameters[f'whitelist_{category}'] = json.load(f)
        return parameters

    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = load_experiments(file_path)

    #################### EXPERT RULES #######################
    domain = 'medical'
    technique = 'rules'
    method = 'expert'
    id = get_id(experiments, domain, technique, method)
    params = {
        'structure': 'nested-class',
        'keywords': get_keywords(domain, method)
    }
    new_row = pd.DataFrame([[id, domain, technique, method, params, None, None, None]],
                           columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    experiments = pd.concat([experiments, new_row], ignore_index=True)

    #################### LLM RULES #######################

    for domain in ['medical', 'administrative', 'education']:
        technique = 'rules'
        method = 'llm_generated'
        id = get_id(experiments, domain, technique, method)
        params = {
            'structure': STRUCTURE_MAPPING[domain][-1], # In LLM Rule generation we specify it is a nested problem
            'token_num': 5,
            'keywords': get_keywords(domain, method)
        }
        new_row = pd.DataFrame([[id, domain, technique, method, params, None, None, None]],
                               columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
        experiments = pd.concat([experiments, new_row], ignore_index=True)
    experiments.to_csv("./results/experiments.csv", index=False)


def rulegen_orchestrator(trains, validations):
    experiments_path = './results/experiments.csv'
    if os.path.exists(experiments_path):
        os.remove(experiments_path)
    rule_loader() # Loads existing rules
    train_rules(trains, validations) # Trains keywords on training set and optimizes token number on validation set. Already done on experiment generation as it is lightweight training

############################################################# NLI ######################################################
def nli_orchestrator(): # Generates the experiments for NLI
    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = load_experiments(file_path)

    for domain in ['medical', 'administrative', 'education']:
        technique = 'nli'
        for structure in STRUCTURE_MAPPING[domain]:
            if structure == 'nested-class':
                prompt = "This hospital is a {}" # Special case for nested classification
            else:
                prompt = "This organization is a {}"                                # get_prompts(trains[domain], classes) Deprecated: We only use one prompt as one/few shot do not work
            for method in ["0_shot"]:                                               #Deprecated: ["1_shot", "few_shot"]: One and Few shot do not work in NLI prompting
                for model in NLI_MODELS.keys():
                    id = get_id(experiments, domain, technique, method)
                    params = {
                        'structure': structure,
                        'model': model,
                        'prompt' : prompt
                    }
                    new_row = pd.DataFrame([[id, domain, technique, method, params, None, None, None]], 
                                          columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                    experiments = pd.concat([experiments, new_row], ignore_index=True)
    experiments.to_csv(file_path, index=False)



############################################################# EMBEDDIGS ######################################################
            ############## Cosine Similarity #################
def similarity_orchestrator(trains, validations, euhub=False):
    file_path = "./results/experiments.csv"
    embeddings_base_path='./results/embeddings/'
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(
            columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = load_experiments(file_path)
    
    # Load label embeddings
    labels_path = os.path.join(embeddings_base_path, 'label_embeddings.csv')
    label_embeddings = load_embeddings(file_path=labels_path)

    for model in EMB_MODELS.keys(): # models are first loop as they are heavy to load
        # Determine embeddings paths based on dataset type
        if euhub:
            embeddings_path = os.path.join(embeddings_base_path, f"euhub_{model}_embeddings.csv") 
        else:
            embeddings_path = os.path.join(embeddings_base_path, f"{model}_embeddings.csv")
            
        # Load embeddings once per model
        embedding_column = f"{model}_embedding"
        print(f"Loading embeddings for model: {model}")
        instance_embeddings = load_embeddings(file_path=embeddings_path)
        
        for domain in ['medical', 'administrative', 'education']:
            technique = 'embedding'
            
            # Merge embeddings with training and validation data
            train = trains[domain].merge(
                instance_embeddings[['instance', embedding_column]], 
                on='instance', 
                how='left'
            )
            
            validation = validations[domain].merge(
                instance_embeddings[['instance', embedding_column]], 
                on='instance', 
                how='left'
            )
            
            # Report missing embeddings
            train_missing = train[embedding_column].isna().sum()
            if train_missing > 0:
                print(f"Warning: {train_missing}/{train[embedding_column].shape[0]} missing embeddings in {domain} domain ")
                
            val_missing = validation[embedding_column].isna().sum()
            if val_missing > 0:
                print(f"Warning: {val_missing}/{validation[embedding_column].shape[0]} missing embeddings in {domain} domain ")

            method = 'similarity'
            for n_shot in ['0_shot', '1_shot', 'few_shot']:
                id = get_id(experiments, domain, technique, method)
                prototypes = {}
                if n_shot == '0_shot':
                    for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                        matching_row = label_embeddings[label_embeddings['label'] == cls.replace('_', ' ')]
                        
                        embedding_value = matching_row[embedding_column].values[0]
                        # Ensure it's a 2D array for cosine_similarity
                        if isinstance(embedding_value, np.ndarray) and embedding_value.ndim == 1:
                            embedding_value = embedding_value.reshape(1, -1)
                            prototypes[cls] = embedding_value
                        else:
                            print(f"WARNING: No embedding found for class '{cls}'")
                            continue  # Skip this class
                elif n_shot == '1_shot':
                    for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                        sampled_id = train[train[cls]==1]['instance'].sample(n=1, random_state=42).values[0]
                        embedding_value = train[train['instance'] == sampled_id][embedding_column].values[0]
                        # Ensure it's a 2D array
                        if isinstance(embedding_value, np.ndarray) and embedding_value.ndim == 1:
                            embedding_value = embedding_value.reshape(1, -1)
                        prototypes[cls] = embedding_value

                elif n_shot == 'few_shot':
                    for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                        prototypes[cls] = {}
                        for country in train['country'].unique():
                            try:
                                sampled_id = train[(train[cls] == 1) & (train['country'] == country)]['instance'].sample(n=1, random_state=42).values[0]
                                embedding_value = train[train['instance'] == sampled_id][embedding_column].values[0]
                                # Ensure it's a 2D array
                                if isinstance(embedding_value, np.ndarray) and embedding_value.ndim == 1:
                                    embedding_value = embedding_value.reshape(1, -1)
                                prototypes[cls][country] = embedding_value
                            except ValueError:
                                # Skip if no matching instances for this country
                                print(f"No instances of class {cls} for country {country}")
                                continue

                validation_exps = pd.DataFrame(
                    columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                for distance in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]:
                    params = {
                        'structure': STRUCTURE_MAPPING[domain][0], # Cosine Similarity has no support for nested classification
                        'n_shot': n_shot,
                        'model': model,
                        'distance': distance,
                        'prototypes': prototypes
                    }
                    val_new_row = pd.DataFrame([[id, domain, technique, method, params, None, None, None]],
                                               columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                    validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)
                best_exp = optimize_parameter(validation_exps, validation, 'distance')  # Get best experiment
                experiments = pd.concat([experiments, best_exp], ignore_index=True)  # Append using concat instead of direct assignment
                experiments.to_csv(file_path, index=False)

            ##############  Classifier Head #################
def classifier_orchestrator(trains, validations, euhub=False, overwrite=False, models=None, classifiers=None):
    file_path = "./results/experiments.csv"
    embeddings_base_path='./results/embeddings/'
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(
            columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = load_experiments(file_path)
    
    # Use all models if none are specified
    if models is None:
        models_to_use = EMB_MODELS.keys()
    else:
        models_to_use = [m for m in models if m in EMB_MODELS]
        if not models_to_use:
            print("Warning: No valid models specified. Using all models.")
            models_to_use = EMB_MODELS.keys()
    
    # Define valid classifier types
    valid_classifiers = ['logreg', 'svm']
    # Use all classifiers if none are specified
    if classifiers is None:
        classifiers_to_use = valid_classifiers
    else:
        classifiers_to_use = [c for c in classifiers if c in valid_classifiers]
        if not classifiers_to_use:
            print("Warning: No valid classifiers specified. Using all classifiers.")
            classifiers_to_use = valid_classifiers

    for model in models_to_use:
        # Determine embeddings paths based on dataset type
        if euhub:
            embeddings_path = os.path.join(embeddings_base_path, f"euhub_{model}_embeddings.csv") 
        else:
            embeddings_path = os.path.join(embeddings_base_path, f"{model}_embeddings.csv")
            
        # Load embeddings once per model
        embedding_column = f"{model}_embedding"
        print(f"Loading embeddings for model: {model}")
        instance_embeddings = load_embeddings(file_path=embeddings_path)

        for domain in ['medical', 'administrative', 'education']:
            technique = 'embedding'
            structure = STRUCTURE_MAPPING[domain][-1] # We will not compare 3-class structure for classifiers.
            train = trains[domain].merge(
                instance_embeddings[['instance', embedding_column]], 
                on='instance', 
                how='left'
            )
            validation = validations[domain].merge(
                instance_embeddings[['instance', embedding_column]], 
                on='instance', 
                how='left'
            )
            method = 'classifier'
            classifier_params = {
                'logreg': [
                    {'C': C, 'solver': solver, 'penalty': 'l1' if solver == 'liblinear' else 'l2'}
                    for C in [0.01, 0.1, 1, 10]
                    for solver in ['liblinear', 'lbfgs']
                ],
                'svm': [
                    {'C': C, 'kernel': kernel, **({'gamma': gamma} if kernel == 'rbf' else {})}
                    for C in [0.1, 1, 10]
                    for kernel in ['linear', 'rbf']
                    for gamma in (['scale', 'auto'] if kernel == 'rbf' else [])
                ]
            }
            for classifier in classifiers_to_use:
                # Check if an experiment with the same configuration already exists
                existing_experiment = False
                if not overwrite:
                    for _, exp in experiments.iterrows():
                        if (exp['Domain'] == domain and 
                            exp['Technique'] == technique and 
                            exp['Method'] == method and 
                            exp['Parameters'].get('model') == model and 
                            exp['Parameters'].get('structure') == structure and 
                            exp['Parameters'].get('classifier') == classifier):
                            
                            existing_experiment = True
                            print(f"Skipping: Experiment already exists for {domain}, {model}, {structure}, {classifier}")
                            
                            # Check if the experiment failed (has no accuracy score)
                            if pd.isna(exp['Accuracy']) or exp['Accuracy'] == 0:
                                print(f"But existing experiment appears to have failed. Set overwrite=True to retry.")
                            break
                if existing_experiment:
                    continue
                    
                # Create validation experiments dataframe
                id = get_id(experiments, domain, technique, method)
                validation_exps = pd.DataFrame(
                    columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                
                # Only use configs for the current classifier
                for config in classifier_params[classifier]:
                    params = {
                        'structure': structure,
                        'model': model,
                        'classifier': classifier,
                        **config
                    }
                    val_new_row = pd.DataFrame([[id, domain, technique, method, params, None, None, None]],
                                            columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                    validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)
                
                # Train models for all configurations
                for idx, exp in tqdm(validation_exps.iterrows(), total=len(validation_exps), desc=f"Training {classifier} for {structure}"):
                    try:
                        trained_model = train_classifier(train, exp)
                        params = exp['Parameters']
                        if structure == 'nested-class' and isinstance(trained_model, dict):
                            for name, clf in trained_model.items():
                                clf_path = save_trained_model(clf, f"{exp['ID']}_{name}", params)
                                params[f"trained_classifier_{name}"] = clf_path
                        else:
                            clf_path = save_trained_model(trained_model, exp['ID'], params)
                            params["trained_classifier"] = clf_path
                        
                        exp = evaluate_classifier_experiment(exp, validation, trained_model) # We do not use the optimize_parameter function to not load all trained models
                        validation_exps.at[idx, 'Accuracy'] = exp['Accuracy']
                        validation_exps.at[idx, 'Recall'] = exp['Recall']
                        validation_exps.at[idx, 'F1'] = exp['F1']
                        validation_exps.at[idx, 'Parameters'] = params

                    except Exception as e:
                        print(f"Error training model for {domain}, structure {structure}, {classifier}: {e}")
                        # Skip this experiment
                        validation_exps = validation_exps.drop(idx)

                if not validation_exps.empty:
                    best_exp_df = validation_exps.loc[[validation_exps['F1'].idxmax()]] 
                    experiments = pd.concat([experiments, best_exp_df], ignore_index=True)
                    experiments.to_csv(file_path, index=False)
                else:
                    print(f"Warning: No valid experiments for {domain}, structure {structure}, {classifier}")

############################## FEW SHOT LEARNING #################################

#
# for model_name in SETFIT_MODELS.keys()
# model = SetFitModel.from_pretrained(
#     model_name,
#     labels=["negative", "positive"],
# )
