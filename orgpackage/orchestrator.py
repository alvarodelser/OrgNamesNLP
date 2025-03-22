import json
import os

import pandas as pd


from orgpackage.config import STRUCTURE_MAPPING, NLI_MODELS, DOMAIN_CLASSES_CORR

from orgpackage.trainer import train_rules


def get_id(experiments, domain, technique, method):
    condition = (experiments['Domain'] == domain) & (experiments['Technique'] == technique) & (experiments['Method'] == method)
    number = len(experiments.loc[condition, 'Parameters'])
    return domain[:3] + '-' + technique[0] + '-' + method.split('_')[0] + '-' + str(number)

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
        experiments = pd.read_csv("./results/experiments.csv")

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
def llm_prompter(trains): # Generates the experiments for NLI
    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
    else:
        experiments = pd.read_csv(file_path)

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
