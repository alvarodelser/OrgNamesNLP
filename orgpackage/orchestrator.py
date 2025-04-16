import json
import os
import ast
import numpy as np
import pandas as pd

from orgpackage.aux import load_experiments, get_id
from orgpackage.config import STRUCTURE_MAPPING, NLI_MODELS, DOMAIN_CLASSES_CORR, EMB_MODELS

from orgpackage.trainer import train_rules, optimize_parameter




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
    new_row = pd.DataFrame([[id, domain, technique, method, params]],
                           columns=["ID", "Domain", "Technique", "Method", "Parameters"])
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
        new_row = pd.DataFrame([[id, domain, technique, method, params]],
                               columns=["ID", "Domain", "Technique", "Method", "Parameters"])
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
                    new_row = pd.DataFrame([[id, domain, technique, method, params]], columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                    experiments = pd.concat([experiments, new_row], ignore_index=True)
    experiments.to_csv(file_path, index=False)



############################################################# EMBEDDIGS ######################################################
def embedding_orchestrator(trains, validations, euhub = False):
    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(
            columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = load_experiments(file_path)



    for model in EMB_MODELS.keys(): # models are first loop as they are heavy to load
        if euhub == True:  # Wikidata dataset
            embeddings = pd.read_csv(f'./results/embeddings/{model}_embeddings.csv')
        else:  # EU Contract Hub Dataset
            embeddings = pd.read_csv(f'./results/embeddings/euhub_{model}_embeddings.csv')
        embeddings = embeddings[['instances', model + '_embedding']]

        for domain in ['medical', 'administrative', 'education']:
            technique = 'embedding'
            train = trains[domain]
            train = train.merge(embeddings, on='instances', how='left')

            validation = validations[domain]
            validation = validation.merge(embeddings, on='instances', how='left')

            ############## Cosine Similarity #################
            method = 'similarity'
            for n_shot in ['0_shot', '1_shot', 'few_shot']:
                id = get_id(experiments, domain, technique, method)
                prototypes = {}
                if n_shot == '0_shot':
                    for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                        label_embeddings = pd.read_csv('results/embeddings/label_embeddings.csv')
                        prototypes[cls] = label_embeddings[label_embeddings['label'] == cls][model + '_embedding'].values[0] # We search in a pre-made list of labels as we cannot store all (fasttext specially) models on memory at the same time.
                elif n_shot == '1_shot':
                    for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                        sampled_id = train[train[cls]==1]['instances'].sample(n=1, random_state=42).values[0]
                        prototypes[cls] = train[train['instances'] == sampled_id][model + '_embedding'].values[0]

                elif n_shot == 'few_shot':
                    for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                        prototypes[cls] = {}
                        for country in train['country'].unique():
                            sampled_id = train[(train[cls] == 1) & (train['country'] == country)]['instances'].sample(n=1, random_state=42).values[0]
                            prototypes[cls][country] = train[train['instances'] == sampled_id][model + '_embedding'].values[0]

                validation_exps = pd.DataFrame(
                    columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                for distance in np.arange(0.1, 1.0, 0.1):
                    params = {
                        'structure': STRUCTURE_MAPPING[domain][0], # Cosine Similarity has no support for nested classification
                        'n_shot': n_shot,
                        'model': model,
                        'distance': distance,
                        'prototypes': prototypes
                    }
                    val_new_row = pd.DataFrame([[id, domain, technique, method, params]],
                                               columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                    validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)
                experiments.loc[len(experiments)] = optimize_parameter(validation_exps, validation, 'distance') # Append only best experiment

        ###############  Classifier Head #################
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

        for model in EMB_MODELS.keys():
            for classifier in classifier_params.keys():
                id = get_id(experiments, domain, technique, method)
                for config in classifier_params[classifier]:
                    validation_exps = pd.DataFrame(
                        columns=["ID", "Domain", "Technique", "Method", "Parameters"])

                    params = {
                        # 'structure': STRUCTURE_MAPPING_WHICH_ONE[domain],
                        'model': model,
                        'classifier': classifier,
                        **config
                    }
                    val_new_row = pd.DataFrame([[id, domain, technique, method, params]],
                                               columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                    validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)



############################## FEW SHOT LEARNING #################################

#
# for model_name in SETFIT_MODELS.keys()
# model = SetFitModel.from_pretrained(
#     model_name,
#     labels=["negative", "positive"],
# )
