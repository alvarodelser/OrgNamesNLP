import json
import ast
import os
import pandas as pd
import gc

import sklearn
from sklearn.metrics import accuracy_score, recall_score, f1_score
from transformers import pipeline, AutoTokenizer, AutoModel

from orgpackage.dataprocesser import load_dataset
from orgpackage.ruleclassifier import country_word_generator, rule_classify
from orgpackage.nliclassifier import avg_accuracy_score, get_prompts, llm_classify
from orgpackage.clusterembedder import get_detailed_instruct, embedder, compute_similarity

DOMAIN_CLASSES_CORR = {
        'medical': ['hospital', 'university_hospital'],
        'administrative': ['local_government'],
        'education': ['primary_school', 'secondary_school']
    }


def get_id(experiments, domain, technique, method):
    condition = (experiments['Domain'] == domain) & (experiments['Technique'] == technique) & (experiments['Method'] == method)
    number = len(experiments.loc[condition, 'Experiment'])
    return domain[:3] + '-' + technique[0] + '-' + method.split('_')[0] + '-' + str(number)


def rule_loader():
    def get_parameters(domain, method):
        parameters = {}
        for category in DOMAIN_CLASSES_CORR.get(domain, []):
            file_name = f'./keywords/{method}/{category}_whitelist.json'
            print(file_name)
            with open(file_name, 'r') as f:
                parameters[f'whitelist_{category}'] = json.load(f)
        return parameters

    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["Domain", "Technique", "Method", "Experiment", "Accuracy", "Recall", "F1"])
    else:
        experiments = pd.read_csv("./results/experiments.csv")

    #################### EXPERT RULES #######################
    domain = 'medical'
    technique = 'rules'
    method = 'expert'
    exp = {
        'id': get_id(experiments, domain, technique, method),
        'structure': 'nested-class',
        'parameters': get_parameters(domain, method),
    }
    experiments.loc[len(experiments)] = [domain, technique, method, exp, None, None, None]

    #################### LLM RULES #######################
    structure_mapping = {
        'medical': 'nested-class',
        'administrative': '2-class',
        'education': '2-multiclass'
    }
    for domain in ['medical', 'administrative', 'education']:
        technique = 'rules'
        method = 'llm_generated'
        exp = {
            'id': get_id(experiments,domain, technique, method),
            'structure': structure_mapping[domain],
            'token_num': 5,
            'parameters': get_parameters(domain, method),
        }
        experiments.loc[len(experiments)] = [domain, technique, method, exp, None, None, None]
    experiments.to_csv("./results/experiments.csv", index=False)


def rule_extractor(train, domain):
    file_path = "./results/experiments.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["Domain", "Technique", "Method", "Experiment", "Accuracy", "Recall", "F1"])
    else:
        experiments = pd.read_csv(file_path)

    #################### MEDICAL DOMAIN #######################
    if domain == 'medical':
        technique = 'rules'
        for method in ['counter_algorithm', 'idf_best']:
            # Hospital 3-class
            exp_id = get_id(experiments, domain, technique, method)
            exp = {
                'id':  exp_id,
                'structure': '3-class',
                'tokenize': False,
                'token_num':5,
                'parameters': {'whitelist_hospital': country_word_generator(train['names'], train['country'], train['hospital'], method),
                               'whitelist_university_hospital': country_word_generator(train['names'], train['country'], train['university_hospital'], method)}
            }
            experiments.loc[len(experiments)] = [domain, technique, method, exp, None, None, None]

            # Hospital Nested 2x 2-class classification
            nested_train = train[train['hospital']==1]
            exp_id = get_id(experiments, domain, technique, method)
            exp = {
                'id': exp_id,
                'structure': 'nested-class',
                'tokenize': False,
                'token_num':5,
                'parameters': {'whitelist_hospital': country_word_generator(train['names'], train['country'], train['hospital'], method), # Same as 2-class
                               'whitelist_university_hospital': country_word_generator(nested_train['names'], nested_train['country'], nested_train['university_hospital'], method)}
            }
            experiments.loc[len(experiments)] = [domain, technique, method, exp, None, None, None]

    #################### ADMINISTRATIVE DOMAIN #######################
    elif domain == 'administrative':
        technique = 'rules'
        for method in ['counter_algorithm', 'idf_best']:
            # Government 2-class classification
            exp_id = get_id(experiments, domain, technique, method)
            exp = {
                'id': exp_id,
                'structure': '2-class',
                'tokenize': False,
                'token_num':5,
                'parameters': {'whitelist_local_government': country_word_generator(train['names'], train['country'], train['local_government'], method)}
            }
            experiments.loc[len(experiments)] = [domain, technique, method, exp, None, None, None]

    #################### EDUCATION DOMAIN #######################
    elif domain == 'education':
        technique = 'rules'
        for method in ['counter_algorithm', 'idf_best']:
            exp_id = get_id(experiments, domain, technique, method)
            exp = {
                'id': exp_id,
                'structure': '2-multiclass',
                'tokenize': False,
                'token_num':5,
                'parameters': {
                    'whitelist_primary_school': country_word_generator(train['names'], train['country'], train['primary_school'], method),
                    'whitelist_secondary_school': country_word_generator(train['names'], train['country'], train['secondary_school'], method)}
            }
            experiments.loc[len(experiments)] = [domain, technique, method, exp, None, None, None]

    experiments.to_csv(file_path, index=False)


def rule_classifier(exp, test):
    classes = DOMAIN_CLASSES_CORR[exp['Domain']]
    pred_columns = []
    for cls in classes:
        col_name = exp['Experiment']['id'] + '_' + cls
        pred_columns.append(col_name)
        test[col_name] = 0
        keywords = exp['Experiment']['parameters']['whitelist_' + cls]

        ### SPECIAL CASES: NESTED CLASSIFICATION OF UNIVERSITY HOSPITALS
        if exp['Experiment'].get('structure', '') == 'nested-class' and cls =='university_hospital':
            test_nested = test[test['hospital'] == 1]
            for country in test['country'].unique():
                test_nested_country = test_nested[test_nested['country'] == country]
                classified_results = rule_classify(test_nested_country['names'], keywords.get(country,[]))
                test.loc[test_nested_country.index, col_name] = classified_results

        for country in test['country'].unique():
            test_country = test[test['country'] == country]  # Subset dataframe
            classified_results = rule_classify(test_country['names'], keywords.get(country,[]))  # Get classification results
            test.loc[test_country.index, col_name] = classified_results

    y_true = test[classes]  # Ground truth columns
    y_pred = test[pred_columns]  # Predicted classification columns
    exp['Accuracy'] = accuracy_score(y_true, y_pred)
    exp['Recall'] = recall_score(y_true, y_pred, average='macro')  # Macro for balanced class importance
    exp['F1'] = f1_score(y_true, y_pred, average='macro')  # Macro-F1 to evaluate per class

    ### COUNTRY-LEVEL METRICS
    exp['Experiment']['country_accuracy'] = {}
    exp['Experiment']['country_recall'] = {}
    exp['Experiment']['country_f1'] = {}
    for country in test['country'].unique():
        test_country = test[test['country'] == country]
        y_true_c = test_country[classes]
        y_pred_c = test_country[pred_columns]

        # Check if all columns in y_true_c
        missing_classes = (y_true_c.sum(axis=0) == 0)
        if test_country.empty or missing_classes.any():
            exp['Experiment']['country_accuracy'][country] = accuracy_score(y_true_c, y_pred_c)
            exp['Experiment']['country_f1'][country] = None
            exp['Experiment']['country_recall'][country] = None
        else:
            exp['Experiment']['country_accuracy'][country] = accuracy_score(y_true_c, y_pred_c)
            exp['Experiment']['country_recall'][country] = recall_score(y_true_c, y_pred_c, average='macro')
            exp['Experiment']['country_f1'][country] = f1_score(y_true_c, y_pred_c, average='macro')


def rulegen_orchestrator(trains):
    experiments_path = './results/experiments.csv'
    if os.path.exists(experiments_path):
        os.remove(experiments_path)
    rule_loader()
    for domain, train_dom in trains.items(): #Extract rules per domain
        rule_extractor(train_dom, domain)


def ruleeval_orchestrator(tests):
    experiments_path = './results/experiments.csv'
    experiments = pd.read_csv(experiments_path)
    experiments['Experiment'] = experiments['Experiment'].apply(ast.literal_eval)
    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'rules':
            rule_classifier(exp, tests[exp['Domain']])  # Modifies the original DataFrame
            experiments.loc[index] = exp
    experiments.to_csv(experiments_path, index=False)

############################################################# NLI ######################################################
def llm_prompter(trains):
    file_path = "./results/experiments_devllm.csv"
    if not os.path.exists(file_path):
        experiments = pd.DataFrame(columns=["Domain", "Technique", "Method", "Experiment", "Accuracy", "Recall", "F1"])
    else:
        experiments = pd.read_csv(file_path)

    structure_mapping = {
        'medical': 'nested-class',
        'administrative': '2-class',
        'education': '2-multiclass'
    }

    for domain in ['medical', 'administrative', 'education']:
        classes = DOMAIN_CLASSES_CORR[domain]
        technique = 'nli'
        prompts = get_prompts(trains[domain], classes)
        for method in ["0_shot", "1_shot", "few_shot"]:
            for model in ["roberta-large", "bge-m3", "mDeBerta", "MiniLM"]:
                exp = {
                    'id': get_id(experiments, domain, technique, method),
                    'structure': structure_mapping[domain],
                    'parameters': {
                        'model': model,
                        'prompt' : prompts[method]
                    }
                }
                new_row = pd.DataFrame([[domain, technique, method, exp]], columns=["Domain", "Technique", "Method", "Experiment"])
                experiments = pd.concat([experiments, new_row], ignore_index=True)
    experiments.to_csv(file_path, index=False)


def llm_classifier(exp, test, classifier, prompt):
    classes = DOMAIN_CLASSES_CORR[exp['Domain']]
    is_multiclass = (exp['Experiment']['structure'] == '2-multiclass')
    print(exp['Domain']+"_"+exp['Experiment']['parameters']['model'])
    results = llm_classify(zero_shot_classifier=classifier,
                            prompt=prompt,
                            names=test['names'],
                            labels = classes.copy(), # Others label is added at llm_classify
                            multi_label = is_multiclass)
    pred_columns = []
    for cls in classes:
        col_name = exp['Experiment']['id'] + '_' + cls
        pred_columns.append(col_name)
        test[col_name] = results[cls]['classification']
        test[col_name + "_conf"] = results[cls]['confidence']

    y_true = test[classes]  # Ground truth columns
    y_pred = test[pred_columns]  # Predicted classification columns
    if is_multiclass:
        exp['Accuracy'] = avg_accuracy_score(y_true, y_pred)
    else:
        exp['Accuracy'] = accuracy_score(y_true, y_pred)
    exp['Recall'] = recall_score(y_true, y_pred, average='macro')  # Macro for balanced class importance
    exp['F1'] = f1_score(y_true, y_pred, average='macro')  # Macro-F1 to evaluate per class


def nlieval_orchestrator(tests):
    models_map = {
        "roberta-large": "joeddav/xlm-roberta-large-xnli",
        "bge-m3": "MoritzLaurer/bge-m3-zeroshot-v2.0",
        "mDeBerta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "MiniLM": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    }
    experiments_path = './results/experiments_devllm.csv'
    experiments = pd.read_csv(experiments_path)
    experiments['Experiment'] = experiments['Experiment'].apply(ast.literal_eval)

    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'nli' and pd.isna(exp['Accuracy']):
            domain = exp['Domain']
            model_key = exp['Experiment']['parameters']['model']
            prompt = exp['Experiment']['parameters']['prompt']
            test_path = "./results/confidences/" + exp['Domain'] + "_" + model_key + ".csv"

            print(f"Loading model: {model_key}")
            model = pipeline("zero-shot-classification", model=models_map[model_key])
            llm_classifier(exp, tests[domain], model, prompt)

            tests[domain].to_csv(test_path, index=False) #SAVING RESULTS PER DOMAIN
            experiments.loc[index] = exp
            experiments.to_csv(experiments_path, index=False) #SAVING PER MODEL

            # Explicitly delete the model and free memory
            del model
            gc.collect()  # Run garbage collection

############################################################# EMBEDDIGS ######################################################


def embedding_orchestrator(tests):
    models_map = {
        'multilingual-e5': {'model_name': 'intfloat/multilingual-e5-large-instruct', 'max_length': 512},
        'qwen': {'model_name': 'Alibaba-NLP/gte-Qwen2-7B-instruct', 'max_length': 8192},
        'mistral': {'model_name': 'Linq-AI-Research/Linq-Embed-Mistral', 'max_length': 4096}
    }
    experiments_path = './results/experiments_embeddings.csv'
    experiments = pd.read_csv(experiments_path)
    experiments['Experiment'] = experiments['Experiment'].apply(ast.literal_eval)

    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'clustering' and pd.isna(exp['Accuracy']):
            domain = exp['Domain']
            classes = DOMAIN_CLASSES_CORR[domain]
            model_key = exp['Experiment']['parameters']['model']
            model_name = models_map[model_key]['model_name']
            max_length = models_map[model_key]['max_length']
            test_path = "./results/similarities/" + exp['Domain'] + "_" + model_key + ".csv"

            # Load from available embedings
            # test_embeddings = generate_embeddings(tests[domain], tokenizer, model, max_length)
            # column_name = exp['Experiment']['id']
            # class_embeddings = generate_embeddings(classes, tokenizer, model, max_length)
            # for idx, cls in enumerate(classes):
            #     cls_embedding = class_embeddings[idx]
            #     tests[domain][column_name + '_' + cls] = compute_similarity(test_embeddings, cls_embedding)
            # tests[domain][column_name + '_embeddings'] = test_embeddings.tolist()
            #
            #
            # tests[domain].to_csv(test_path, index=False)

















            experiments.loc[index] = exp
            experiments.to_csv(experiments_path, index=False)  # SAVING PER MODEL
            #
            # # Free memory
            # del tokenizer
            # del model
            # gc.collect()  # Run garbage collection
            #
            #
            # model_name = exp['Experiment']['parameters']['model']
            # embed_and_save(exp, tests, model_name)
            #
            # experiments.at[index, 'Processed'] = True
            # experiments.to_csv(experiments_path, index=False)
            #
            # gc.collect()

#
# def clustering_vectorizer():
#     FAST TEXT
#     EMBEDDING MODELS = ()
#     SENTENCE EMBEDDINGS MODELS = ()
#
# def clustering_classifier():
#     DBSCAN
#     HIERARCHICAL
#     CLOSEST LABELS
#     LABEL ASSIGNMENT
#