import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from orgpackage.config import STRUCTURE_MAPPING, DOMAIN_CLASSES_CORR, EMB_MODELS
from orgpackage.aux import load_experiments, get_id
from orgpackage.ruleclassifier import country_word_generator
from orgpackage.evaluator import evaluate_rule_experiment, evaluate_cosine_experiment




def optimize_parameter(validation_exps, validation, parameter): #token_num, distance
    file_path = f"./results/parameter_optimization/{parameter}.csv"
    if not os.path.exists(file_path):
        param_exps = pd.DataFrame(
            columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        param_exps = pd.read_csv(file_path)
    for index, exp in tqdm(validation_exps.iterrows(), total=len(validation_exps), desc=f"Optimizing {parameter}"):
        if parameter == 'token_num':
            exp = evaluate_rule_experiment(exp, validation)  # Evaluation on validation set
        elif parameter == 'distance':
            exp = evaluate_cosine_experiment(exp, validation)
        validation_exps.loc[index] = exp
    best_exp = validation_exps.loc[validation_exps['F1'].idxmax()]

    # Saving optimization results
    param_exps = pd.concat([param_exps, validation_exps], ignore_index=True)
    param_exps.to_csv(file_path, index=False)

    return best_exp

def train_rules(trains, validations):
    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = load_experiments(file_path)

    for domain in ['medical', 'administrative', 'education']:
        train = trains[domain]
        validation = validations[domain]

        technique = 'rules'
        for method in ['counter_algorithm', 'idf_best']:
            for structure in STRUCTURE_MAPPING[domain]:
                for preprocessing, name_col in [('None', 'names'), ('Spacy tokenization', 'tokenized'),
                                                ('Decomposition', 'decomposed')]:
                    id = get_id(experiments, domain, technique, method)
                    validation_exps = pd.DataFrame(columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                    for token_number in [3,4,5,6,7,8,9,10]:
                        keywords = {}
                        for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                            if structure == 'nested-class' and cls == 'university_hospital': # Special nested training on only hospital data
                                train_hospitals = train[train['hospital']==1]
                                keywords['whitelist_' + cls] = country_word_generator(train_hospitals[name_col], train_hospitals['country'],
                                                                                      train_hospitals[cls], method, token_number)
                            else:
                                keywords['whitelist_'+cls] = country_word_generator(train[name_col], train['country'], train[cls], method, token_number)
                        params = {
                            'structure': structure,
                            'preprocessing': preprocessing,
                            'token_num': token_number,
                            'keywords': keywords
                        }
                        val_new_row = pd.DataFrame([[id, domain, technique, method, params]],
                                           columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                        validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)

                    new_row = optimize_parameter(validation_exps, validation, 'token_num')
                    print(new_row)
                    experiments = pd.concat([experiments, new_row], ignore_index=True)

    experiments.to_csv(file_path, index=False)


############################################################# NLI ######################################################
#No training needed for 0-shot
############################################################# EMBEDDIGS ######################################################
