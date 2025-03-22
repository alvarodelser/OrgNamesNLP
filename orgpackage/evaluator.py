import ast
import gc

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from transformers import pipeline

from orgpackage.config import DOMAIN_CLASSES_CORR, NLI_MODELS
from orgpackage.ruleclassifier import  rule_classify
from orgpackage.nliclassifier import avg_accuracy_score, llm_classify

############################################################# RULES ######################################################
def evaluate_rule_experiment(exp, test):
    classes = DOMAIN_CLASSES_CORR[exp['Domain']]
    pred_columns = []
    for cls in classes:
        keywords = exp['Parameters']['keywords']['whitelist_' + cls]
        tokenize = exp['Parameters']['tokenize']

        col_name = exp['ID'] + '_' + cls
        pred_columns.append(col_name)
        test[col_name] = 0

        for country in test['country'].unique():
            test_country = test[test['country'] == country]

            ### SPECIAL CASE: NESTED CLASSIFICATION OF UNIVERSITY HOSPITALS
            if exp['Parameters'].get('structure', '') == 'nested-class' and cls == 'university_hospital':
                test_country = test_country[test_country['hospital'] == 1]

            rules = keywords.get(country, [])
            if tokenize:
                names = test_country['names']
            else:
                names = test_country['tokenized']

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
    experiments = pd.read_csv(experiments_path)
    experiments['Parameters'] = experiments['Parameters'].apply(ast.literal_eval)
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
    experiments = pd.read_csv(experiments_path)
    experiments['Parameters'] = experiments['Parameters'].apply(ast.literal_eval)

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
def evaluate_embeddings(tests):
    experiments_path = './results/experiments_embeddings.csv'
    experiments = pd.read_csv(experiments_path)
    experiments['Parameters'] = experiments['Parameters'].apply(ast.literal_eval)

    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'vector' and exp['Method'] == 'similarity' and pd.isna(exp['Accuracy']):
            domain = exp['Domain']
            classes = DOMAIN_CLASSES_CORR[domain]
            model_key = exp['Parameters']['model']


            test_path = "./results/similarities/" + exp['Domain'] + "_" + model_key + ".csv"

            # Load from available embedings
            # test_embeddings = generate_embeddings(tests[domain], tokenizer, model, max_length)
            # column_name = exp['ID']
            # class_embeddings = generate_embeddings(classes, tokenizer, model, max_length)
            # for idx, cls in enumerate(classes):
            #     cls_embedding = class_embeddings[idx]
            #similarity_matrix = embeddings @ class_embeddings.T
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
            # model_name = exp['Parameters']['model']
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