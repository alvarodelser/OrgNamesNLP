import ast
import os
import pandas as pd
import numpy as np

from orgpackage.config import STRUCTURE_MAPPING, DOMAIN_CLASSES_CORR, EMB_MODELS
from orgpackage.orchestrator import get_id
from orgpackage.ruleclassifier import country_word_generator
from orgpackage.evaluator import evaluate_rule_experiment




def optimize_token_number(validation_exps, validation):
    file_path = "./results/parameter_optimization/token_num.csv"
    if not os.path.exists(file_path):
        token_num_exps = pd.DataFrame(
            columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        token_num_exps = pd.read_csv(file_path)
    for index, exp in validation_exps.iterrows():
        exp = evaluate_rule_experiment(exp, validation)  # Evaluation on validation set
        validation_exps.loc[index] = exp
    best_exp = validation_exps.loc[validation_exps['F1'].idxmax()]

    # Saving optimization results
    token_num_exps = pd.concat([token_num_exps, validation_exps], ignore_index=True)
    token_num_exps.to_csv("./results/parameter_optimization/token_num.csv", index=False)

    return best_exp


def train_rules(trains, validations):
    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = pd.read_csv(file_path)

    for domain in ['medical', 'administrative', 'education']:
        train = trains[domain]
        validation = validations[domain]
        technique = 'rules'
        for method in ['counter_algorithm', 'idf_best']:
            for structure in STRUCTURE_MAPPING[domain]:
                for preprocessing, name_col in [('None', 'names'), ('SpaCy tokenization', 'tokenized'),
                                                ('Decomposition', 'decomposed')]:
                    id = get_id(experiments, domain, technique, method)
                    validation_exps = pd.DataFrame(columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                    for token_number in [3,4,5,6,7,8,9,10]:
                        keywords = {}
                        for cls in DOMAIN_CLASSES_CORR.get(domain, []):
                            if structure == 'nested-class' and cls == 'university_hospital': # Special nested training on only hospital data
                                train_hospitals = train[train['hospital']==1]
                                keywords['whitelist_' + cls] = country_word_generator(train_hospitals[name_col], train_hospitals['country'],
                                                                                      train_hospitals[cls], method, token_number),
                            else:
                                keywords['whitelist_'+cls] = country_word_generator(train[name_col], train['country'], train[cls], method, token_number),
                        params = {
                            'structure': structure,
                            'preprocessing': preprocessing,
                            'token_num': token_number,
                            'keywords': keywords
                        }
                        val_new_row = pd.DataFrame([[id, domain, technique, method, params]],
                                           columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                        validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)

                    new_row = optimize_token_number(validation_exps, validation)
                    experiments = pd.concat([experiments, new_row], ignore_index=True)

    experiments.to_csv(file_path, index=False)


############################################################# NLI ######################################################
#No training needed for 0-shot
############################################################# EMBEDDIGS ######################################################
def embedding_exp_gen():
    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(
            columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = pd.read_csv(file_path)
    for domain in ['medical', 'administrative', 'education']:
        technique = 'embedding'

        ############## Cosine Similarity #################
        method = 'similarity'
        for model in EMB_MODELS.keys():
            id = get_id(experiments, domain, technique, method)
            for threshold in np.arange(0.1, 1.0, 0.1):
                validation_exps = pd.DataFrame(
                    columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                params = {
                    'structure': STRUCTURE_MAPPING[domain][0], # Cosine Similarity has no support for nested classification
                    'model': model,
                    'threshold': threshold,
                }
                val_new_row = pd.DataFrame([[id, domain, technique, method, params]],
                                           columns=["ID", "Domain", "Technique", "Method", "Parameters"])
                validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)

        #     for training_data in ['0_shot', '1_shot', 'few_shot']:
        #     for threshold in np.range
        #         exp = {
        #             'id': get_id(experiments, domain, technique, method),
        #             'structure': structure_mapping[domain],
        #             'training_data': training_data
        #             'parameters': {
        #                 'threshold': threshold
        #             }
        #         }
        #         experiments.loc[len(experiments)] = [id, domain, technique, method, exp, None, None, None]
        #
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
                        'structure': STRUCTURE_MAPPING_WHICH_ONE[domain],
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
