import ast
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from orgpackage.config import DOMAIN_CLASSES_CORR, NLI_MODELS
from orgpackage.orchestrator import load_experiments
from orgpackage.ruleclassifier import  rule_classify
from orgpackage.nliclassifier import avg_accuracy_score, llm_classify


############################################################# RULES ######################################################
def evaluate_rule_experiment(exp, test):
    classes = DOMAIN_CLASSES_CORR[exp['Domain']]
    pred_columns = []
    for cls in classes:
        keywords = exp['Parameters']['keywords']['whitelist_' + cls]
        preprocessing = exp['Parameters']['preprocessing']

        col_name = exp['ID'] + '_' + cls
        pred_columns.append(col_name)
        test[col_name] = 0

        for country in test['country'].unique():
            test_country = test[test['country'] == country]

            ### SPECIAL CASE: NESTED CLASSIFICATION OF UNIVERSITY HOSPITALS
            if exp['Parameters'].get('structure', '') == 'nested-class' and cls == 'university_hospital':
                test_country = test_country[test_country['hospital'] == 1]

            rules = keywords.get(country, [])
            if preprocessing == 'None':
                names = test_country['names']
            elif preprocessing == 'Spacy tokenization':
                names = test_country['tokenized']
            else:
                names = test_country['decomposed']

            classified_results = rule_classify(names, rules)  # Get classification results
            test.loc[test_country.index, col_name] = classified_results

    y_true = test[classes]  # Ground truth columns
    y_pred = test[pred_columns]  # Predicted classification columns
    exp['Accuracy'] = accuracy_score(y_true, y_pred)
    exp['Recall'] = recall_score(y_true, y_pred, average='macro')  # Macro for balanced class importance
    exp['F1'] = f1_score(y_true, y_pred, average='macro')  # Macro-F1 to evaluate per class

    ### COUNTRY-LEVEL METRICS
    exp['Parameters']['country_accuracy'] = {}
    exp['Parameters']['country_recall'] = {}
    exp['Parameters']['country_f1'] = {}
    for country in test['country'].unique():
        test_country = test[test['country'] == country]
        y_true_c = test_country[classes]
        y_pred_c = test_country[pred_columns]

        # Check if all columns in y_true_c
        missing_classes = (y_true_c.sum(axis=0) == 0)
        if test_country.empty or missing_classes.any():
            exp['Parameters']['country_accuracy'][country] = accuracy_score(y_true_c, y_pred_c)
            exp['Parameters']['country_f1'][country] = None
            exp['Parameters']['country_recall'][country] = None
        else:
            exp['Parameters']['country_accuracy'][country] = accuracy_score(y_true_c, y_pred_c)
            exp['Parameters']['country_recall'][country] = recall_score(y_true_c, y_pred_c, average='macro')
            exp['Parameters']['country_f1'][country] = f1_score(y_true_c, y_pred_c, average='macro')
    return exp



def evaluate_rules(tests):
    experiments_path = './results/experiments.csv'
    experiments = load_experiments(experiments_path)
    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'rules':
            exp = evaluate_rule_experiment(exp, tests[exp['Domain']])  # Modifies the original DataFrame
            experiments.loc[index] = exp
    experiments.to_csv(experiments_path, index=False)


############################################################# NLI ######################################################
def evaluate_nli_experiment(exp, test, classifier, prompt): # Evaluates one model
    classes = DOMAIN_CLASSES_CORR[exp['Domain']]
    is_multiclass = (exp['Parameters']['structure'] == '2-multiclass')
    print(exp['Domain']+"_"+exp['Parameters']['model'])
    results = llm_classify(zero_shot_classifier=classifier,
                            prompt=prompt,
                            names=test['names'],
                            labels = classes.copy(), # Others label is added at llm_classify
                            multi_label = is_multiclass)
    pred_columns = []
    for cls in classes:
        col_name = exp['ID'] + '_' + cls
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
    return exp


def evaluate_nli(tests): # Evaluates per model of experiments
    experiments_path = './results/experiments.csv'
    experiments = load_experiments(experiments_path)

    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'nli' and pd.isna(exp['Accuracy']):
            domain = exp['Domain']
            model_key = exp['Parameters']['model']
            prompt = exp['Parameters']['prompt']
            test_path = "./results/confidences/" + exp['Domain'] + "_" + model_key + ".csv"

            print(f"Loading model: {model_key}")
            model = pipeline("zero-shot-classification", model=NLI_MODELS[model_key])
            exp = evaluate_nli_experiment(exp, tests[domain], model, prompt)

            tests[domain].to_csv(test_path, index=False) #SAVING RESULTS PER DOMAIN
            experiments.loc[index] = exp
            experiments.to_csv(experiments_path, index=False) #SAVING PER MODEL

            # Explicitly delete the model and free memory
            del model
            gc.collect()  # Run garbage collection


############################################################# EMBEDDIGS ######################################################
def evaluate_cosine_experiment(exp, test):
    distance = exp['Parameters']['distance']
    threshold = 1 - distance  # cosine similarity threshold
    prototypes = exp['Parameters']['prototypes']
    model = exp['Parameters']['model']
    embedding_col = model + '_embedding'
    domain = exp['Domain']
    structure = exp['Parameters']['structure']
    is_multiclass = (structure == '2-multiclass')
    classes = DOMAIN_CLASSES_CORR[domain]

    print(f"{domain}_{model}")

    pred_columns = [f"{exp['ID']}_{cls}" for cls in classes]

    for col in pred_columns:
        test[col] = 0  # Initialize predictions

    for idx, row in test.iterrows():
        x = row[embedding_col]
        if isinstance(x, str):
            x = np.array(eval(x))

        similarities = {}

        for cls in classes:
            proto = prototypes.get(cls)
            best_sim = -1

            if isinstance(proto, dict):  # few-shot: multiple country prototypes
                for p in proto.values():
                    if isinstance(p, str):
                        p = np.array(eval(p))
                    sim = cosine_similarity([x], [p])[0][0]
                    best_sim = max(best_sim, sim)
            elif proto is not None:
                if isinstance(proto, str):
                    proto = np.array(eval(proto))
                best_sim = cosine_similarity([x], [proto])[0][0]

            similarities[cls] = best_sim

        if is_multiclass:
            for cls in classes:
                if similarities[cls] >= threshold:
                    test.at[idx, f"{exp['ID']}_{cls}"] = 1
        else:
            # Multiclass: choose the closest class (max sim), if over threshold
            best_cls = max(similarities, key=similarities.get)
            if similarities[best_cls] >= threshold:
                test.at[idx, f"{exp['ID']}_{best_cls}"] = 1  # One-hot

    y_true = test[classes]
    y_pred = test[pred_columns]

    if is_multiclass:
        exp['Accuracy'] = accuracy_score(y_true, y_pred)
    else:
        exp['Accuracy'] = avg_accuracy_score(y_true, y_pred)  # Custom function or define separately

    exp['Recall'] = recall_score(y_true, y_pred, average='macro')
    exp['F1'] = f1_score(y_true, y_pred, average='macro')

    return exp


def evaluate_embeddings(tests):
    file_path = './results/experiments.csv'
    experiments = load_experiments(file_path)
    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'vector' and exp['Method'] == 'similarity':
            domain = exp['Domain']
            experiments.loc[index] = evaluate_cosine_experiment(exp, tests[domain])
    experiments.to_csv(file_path, index=False)
        
