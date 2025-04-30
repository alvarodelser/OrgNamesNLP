import ast
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from tqdm import tqdm
import os # Added for path joining

from orgpackage.config import DOMAIN_CLASSES_CORR, NLI_MODELS, COUNTRY_DICT
from orgpackage.orchestrator import load_experiments
from orgpackage.ruleclassifier import  rule_classify
from orgpackage.aux import load_dataset, load_trained_model, prepare_labels

import ast
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

def avg_accuracy_score(y_true, y_pred):
    accuracy_list = []
    for label in y_pred.columns:  # For each label (column) in the DataFrame
        label_true = y_true[label.split("_",1)[1]]  # The class column is the predictions column minus the experiment id
        label_pred = y_pred[label]  # Predicted values for this label
        label_accuracy = accuracy_score(label_true, label_pred)
        accuracy_list.append(label_accuracy)
    return np.mean(accuracy_list)  # Average accuracy over all samples

############################################################# RULES ######################################################
def evaluate_rule_experiment(exp, test, test_countries=None):
    classes = DOMAIN_CLASSES_CORR[exp['Domain']]
    pred_columns = []
    for cls in classes:
        keywords = exp['Parameters']['keywords']['whitelist_' + cls]
        preprocessing = exp['Parameters'].get('preprocessing', 'None')

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

    if exp['Method'] == 'expert': # Only experts from France, Italy, and Austria answered the questionnaire
        filtered_test = test[test['country'].isin(['Q142', 'Q38', 'Q40'])]
    elif test_countries is not None and exp['Method'] in ['counter_algorithm', 'idf_best']: # For models that need training, we use only countries with sufficient data
        filtered_test = test[test['country'].isin(test_countries)]
    else:
        filtered_test = test.copy()
    
    y_true = filtered_test[classes]  # Ground truth columns
    y_pred = filtered_test[pred_columns]  # Predicted classification columns
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
    data = load_dataset()

    # Determine valid countries per domain based on min class volume
    test_countries = {}
    for domain in DOMAIN_CLASSES_CORR:
        test_countries[domain] = []
        for country in COUNTRY_DICT:
            country_df = data[data['country'] == country]
            class_counts = [
                country_df[country_df[cls] == 1].shape[0]
                for cls in DOMAIN_CLASSES_CORR[domain]
            ]
            min_country_vol = min(class_counts) if class_counts else 0
            if min_country_vol >= 30:
                test_countries[domain].append(country)

    experiments_path = './results/experiments.csv'
    experiments = load_experiments(experiments_path)
    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'rules':
            exp = evaluate_rule_experiment(exp, tests[exp['Domain']], test_countries[exp['Domain']])  # Modifies the original DataFrame
            experiments.loc[index] = exp
    experiments.to_csv(experiments_path, index=False)


############################################################# NLI ######################################################
def nli_classify(zero_shot_classifier, prompt: str, names: list, labels: list, multi_label: bool = False):
    THRESHOLD = 0.65

    if 'other' not in labels:
        labels.append('other')
    if multi_label:
        labels.remove('other')

    results = {label: {"classification": [], "confidence": []} for label in labels}
    for name in tqdm(names, desc="Classifying organizations", unit="organization"):
        result = zero_shot_classifier(name,
                                        labels,
                                        hypothesis_template = prompt,
                                        multi_label=multi_label)
        for label in labels:
            if multi_label:
                if label in result["labels"]:
                    idx = result["labels"].index(label)
                    confidence = result["scores"][idx]
                    is_classified = int(confidence >= THRESHOLD)
                else:
                    confidence = 0.0
                    is_classified = 0
            else:
                top_label = result["labels"][0]
                confidence = result["scores"][0] if label == top_label else 0.0
                is_classified = 1 if label == top_label else 0
            results[label]["classification"].append(is_classified)
            results[label]["confidence"].append(confidence)
    return results


def evaluate_nli_experiment(exp, test, classifier, prompt): # Evaluates one model
    classes = DOMAIN_CLASSES_CORR[exp['Domain']]
    is_multiclass = (exp['Parameters']['structure'] == '2-multiclass')
    print(exp['Domain']+"_"+exp['Parameters']['model'])
    results = nli_classify(zero_shot_classifier=classifier,
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
def evaluate_similarity_experiment(exp, test):
    distance = exp['Parameters']['distance']
    threshold = 1 - distance  # cosine similarity threshold
    prototypes = exp['Parameters']['prototypes']
    model = exp['Parameters']['model']
    embedding_col = model + '_embedding'
    domain = exp['Domain']
    structure = exp['Parameters']['structure']
    is_multiclass = (structure == '2-multiclass')
    classes = DOMAIN_CLASSES_CORR[domain]
    
    pred_columns = [f"{exp['ID']}_{cls}" for cls in classes]
    
    # Add columns for storing similarity scores
    sim_columns = [f"sim_{cls}" for cls in classes]
    for col in sim_columns:
        test[col] = -1.0  # Initialize as float
    
    for col in pred_columns:
        test[col] = 0.0  # Initialize predictions as float
    
    # Count valid embeddings
    valid_embeddings = 0
    total_embeddings = len(test)
    sim_calculation_attempts = 0
    sim_calculation_errors = 0
    
    # Print some diagnostic information about the prototypes
    for cls in classes:
        proto = prototypes.get(cls)
        if proto is None:
            print(f"Missing prototype for class: {cls}")
            continue
            
        if isinstance(proto, dict):
            print(f"Prototype for {cls} is a dictionary with {len(proto)} country entries")
            # Print shape of first country prototype for diagnosis
            for country, p in proto.items():
                if p is not None:
                    if isinstance(p, np.ndarray):
                        print(f"  Country {country} prototype shape: {p.shape}")
                    break
        else:
            if isinstance(proto, np.ndarray):
                print(f"Prototype for {cls} shape: {proto.shape}")
    
    for idx, row in test.iterrows():
        x = row[embedding_col]
        
        # For instances with missing embeddings, keep predictions as 0
        if isinstance(x, (float, int)) and pd.isna(x):
            continue
        elif isinstance(x, np.ndarray) and np.isnan(x).any():
            continue
        elif x is None:
            continue
            
        # Convert string representation to numpy array if needed
        if isinstance(x, str):
            try:
                x = np.array(ast.literal_eval(x))
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing embedding at index {idx}: {e}")
                # Keep predictions as 0
                continue
                
        # Validate numpy array
        if not isinstance(x, np.ndarray):
            print(f"Unexpected embedding type at index {idx}: {type(x)}")
            # Keep predictions as 0
            continue
            
        # Check for NaN values in the array
        if np.isnan(x).any():
            print(f"NaN values detected in embedding at index {idx}")
            # Keep predictions as 0
            continue
            
        # Ensure x is 1D or reshape properly to a 2D array with one row
        if len(x.shape) > 2:
            x = x.reshape(1, -1) if len(x.shape) == 3 and x.shape[0] == 1 else x.flatten()
        elif len(x.shape) == 1:
            x = x.reshape(1, -1)  # Reshape 1D to 2D for cosine_similarity
            
        valid_embeddings += 1
        similarities = {}

        for cls in classes:
            proto = prototypes.get(cls)
            best_sim = -1.0

            # Skip if prototype is missing
            if proto is None:
                similarities[cls] = -1.0
                continue

            if isinstance(proto, dict):  # few-shot: multiple country prototypes
                valid_protos = 0
                for country, p in proto.items():
                    # Skip invalid prototypes
                    if p is None or (isinstance(p, np.ndarray) and pd.isna(p).any()):
                        continue
                        
                    # Convert string to array if needed
                    if isinstance(p, str):
                        try:
                            p = np.array(ast.literal_eval(p))
                        except (ValueError, SyntaxError):
                            continue
                    
                    # # Check for NaN values
                    # if isinstance(p, np.ndarray) and np.isnan(p).any():
                    #     continue
                    
                    # Ensure prototype is properly shaped
                    if len(p.shape) > 2:
                        p = p.reshape(1, -1) if len(p.shape) == 3 and p.shape[0] == 1 else p.flatten()
                    elif len(p.shape) == 1:
                        p = p.reshape(1, -1)  # Reshape 1D to 2D
                        
                    # Calculate similarity
                    try:
                        sim_calculation_attempts += 1
                        sim = cosine_similarity(x, p)[0][0]
                        best_sim = max(best_sim, sim)
                        valid_protos += 1
                    except Exception as e:
                        sim_calculation_errors += 1
                        if sim_calculation_errors <= 5:  # Limit error messages
                            print(f"Error calculating similarity: {e}")
                            print(f"  Embedding shape: {x.shape}, Prototype shape: {p.shape}")
                        continue
                
                if valid_protos == 0:
                    similarities[cls] = -1.0
                else:
                    similarities[cls] = best_sim
            else:
                # Handle single prototype
                # Skip invalid prototypes
                if pd.isna(proto).any():
                    similarities[cls] = -1.0
                    continue
                    
                # Convert string to array if needed
                if isinstance(proto, str):
                    try:
                        proto = np.array(ast.literal_eval(proto))
                    except (ValueError, SyntaxError):
                        similarities[cls] = -1.0
                        continue
                
                # Check for NaN values
                # if isinstance(proto, np.ndarray) and np.isnan(proto).any():
                #     similarities[cls] = -1.0
                #     continue
                
                # Ensure prototype is properly shaped
                if len(proto.shape) > 2:
                    proto = proto.reshape(1, -1) if len(proto.shape) == 3 and proto.shape[0] == 1 else proto.flatten()
                elif len(proto.shape) == 1:
                    proto = proto.reshape(1, -1)  # Reshape 1D to 2D
                    
                # Calculate similarity
                try:
                    sim_calculation_attempts += 1
                    best_sim = cosine_similarity(x, proto)[0][0]
                    similarities[cls] = best_sim
                except Exception as e:
                    sim_calculation_errors += 1
                    if sim_calculation_errors <= 5:  # Limit error messages
                        print(f"Error calculating similarity: {e}")
                        print(f"  Embedding shape: {x.shape}, Prototype shape: {proto.shape}")
                    similarities[cls] = -1.0
                    continue

        # If we have no valid similarities, keep predictions as 0
        if all(sim == -1.0 for sim in similarities.values()):
            continue
            
        # Store similarity scores
        for cls in classes:
            if similarities.get(cls, -1.0) != -1.0:
                test.at[idx, f"sim_{cls}"] = float(similarities[cls])

        if is_multiclass:
            for cls in classes:
                if similarities[cls] >= threshold and similarities[cls] != -1.0:
                    test.at[idx, f"{exp['ID']}_{cls}"] = 1.0
        else:
            # Filter out invalid similarities (-1)
            valid_similarities = {k: v for k, v in similarities.items() if v != -1.0}
            if valid_similarities:
                # Choose the closest class (max sim), if over threshold
                best_cls = max(valid_similarities, key=valid_similarities.get)
                if valid_similarities[best_cls] >= threshold:
                    test.at[idx, f"{exp['ID']}_{best_cls}"] = 1.0  # One-hot

    # Print diagnostic information
    print(f"Valid embeddings: {valid_embeddings} out of {total_embeddings} ({100 * valid_embeddings / total_embeddings:.2f}%)")
    print(f"Similarity calculations: {sim_calculation_attempts} attempts, {sim_calculation_errors} errors ({100 * sim_calculation_errors / max(1, sim_calculation_attempts):.2f}% error rate)")
    
    # # Check thresholds and similarity distributions
    # for cls in classes:
    #     sim_col = f"sim_{cls}"
    #     if sim_col in test.columns:
    #         valid_sims = test[test[sim_col] >= 0][sim_col]
    #         if not valid_sims.empty:
    #             print(f"Similarity stats for {cls}:")
    #             print(f"  Min: {valid_sims.min():.4f}, Max: {valid_sims.max():.4f}, Mean: {valid_sims.mean():.4f}")
    #             print(f"  Current threshold: {threshold:.4f}")
    #             # Calculate what threshold would give ~same positives as true values
    #             true_count = test[cls].sum()
    #             if true_count > 0:
    #                 optimal_threshold = valid_sims.sort_values(ascending=False).iloc[min(true_count, len(valid_sims)-1)]
    #                 print(f"  Suggested threshold for balanced prediction count: {optimal_threshold:.4f}")
    
    # Convert prediction columns to int for evaluation
    for col in pred_columns:
        test[col] = test[col].astype(int)
    
    # Analyze class distribution in true and predicted labels
    for cls in classes:
        true_count = test[cls].sum()
        pred_count = test[f"{exp['ID']}_{cls}"].sum()
        print(f"Class {cls}: true={true_count}, predicted={pred_count}")
    
    y_true = test[classes]
    y_pred = test[pred_columns]

    if is_multiclass:
        exp['Accuracy'] = accuracy_score(y_true, y_pred)
    else:
        exp['Accuracy'] = avg_accuracy_score(y_true, y_pred)  # Custom function or define separately

    # Add per-class metrics for more detailed analysis
    for cls in classes:
        true_cls = test[cls]
        pred_cls = test[f"{exp['ID']}_{cls}"]
        cls_recall = recall_score(true_cls, pred_cls, average=None) if true_cls.sum() > 0 else 0
        exp[f'Recall_{cls}'] = cls_recall
        print(f"Recall for {cls}: {cls_recall}")

    exp['Recall'] = recall_score(y_true, y_pred, average='macro')
    exp['F1'] = f1_score(y_true, y_pred, average='macro')
    
    return exp



def evaluate_classifier_experiment(exp, validation, clf=None):
    import ast
    import numpy as np
    from sklearn.metrics import accuracy_score, recall_score, f1_score

    model = exp['Parameters']['model']
    embedding_col = model + '_embedding'
    domain = exp['Domain']
    structure = exp['Parameters'].get('structure', 'flat')
    classes = DOMAIN_CLASSES_CORR[domain]

    # Parse embeddings
    X, valid_indices = [], []
    for idx, emb in enumerate(validation[embedding_col]):
        try:
            if emb is None or (np.isscalar(emb) and pd.isna(emb)):
                continue
            embedding = np.array(ast.literal_eval(emb)) if isinstance(emb, str) else emb
            if isinstance(embedding, np.ndarray) and not np.isnan(embedding).any():
                X.append(embedding)
                valid_indices.append(idx)
        except:
            continue

    if not X:
        print(f"WARNING: No valid embeddings found for {domain} domain!")
        exp['Accuracy'] = 0
        exp['Recall'] = 0
        exp['F1'] = 0
        return exp

    X = np.array(X)
    filtered_validation = validation.iloc[valid_indices].copy()
    y = prepare_labels(filtered_validation, classes, structure)

    # Load classifier if not provided
    if clf is None:
        try:
            if structure == 'nested-class':
                clf = {
                    'nested_classifier': {
                        name: load_trained_model(exp['Parameters'][f"trained_classifier_{name}"])
                        for name in ['hospital', 'university_hospital']
                        if f"trained_classifier_{name}" in exp['Parameters']
                    },
                    'structure': 'nested-class'
                }
            else:
                clf = load_trained_model(exp['Parameters']['trained_classifier'])
        except Exception as e:
            print(f"ERROR: Failed to load classifier: {e}")
            exp['Accuracy'] = 0
            exp['Recall'] = 0
            exp['F1'] = 0
            return exp

    try:
        if structure == 'nested-class' and isinstance(clf, dict):
            # Access classifiers correctly nested within 'nested_classifier' key
            nested_clfs = clf.get('nested_classifier', {})
            if 'hospital' not in nested_clfs:
                 raise KeyError("Nested classifier dictionary is missing the 'hospital' model.")
            hospital_clf = nested_clfs['hospital']
            y_pred_hosp = hospital_clf.predict(X)

            y_pred = np.zeros((len(X), len(classes)), dtype=int)
            hospital_idx = classes.index('hospital')
            y_pred[:, hospital_idx] = y_pred_hosp

            # Check for university_hospital classifier within the nested dictionary
            if 'university_hospital' in classes and 'university_hospital' in nested_clfs:
                univ_clf = nested_clfs['university_hospital']
                univ_idx = classes.index('university_hospital')
                hosp_indices = np.where(y_pred_hosp == 1)[0]
                if len(hosp_indices) > 0:
                    X_hosp = X[hosp_indices]
                    y_pred_univ = univ_clf.predict(X_hosp) # Use univ_clf
                    y_pred[hosp_indices, univ_idx] = y_pred_univ

                non_hosp_indices = np.where(y_pred_hosp == 0)[0]
                y_pred[non_hosp_indices, univ_idx] = 0
            elif 'university_hospital' in classes:
                # Handle case where university_hospital class exists but model wasn't loaded/found
                print("Warning: 'university_hospital' class exists but its model was not found in the nested classifier dictionary.")
                univ_idx = classes.index('university_hospital')
                y_pred[:, univ_idx] = 0 # Ensure predictions are 0

            print(f"Nested classification: Predicted {np.sum(y_pred[:, hospital_idx])} hospitals and {np.sum(y_pred[:, univ_idx]) if 'university_hospital' in classes else 0} university hospitals")
        else:
            # Standard prediction for non-nested structures
            y_pred = clf.predict(X)
    except Exception as e:
        print(f"ERROR: Failed to make predictions: {e}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        exp['Accuracy'] = 0
        exp['Recall'] = 0
        exp['F1'] = 0
        return exp

    # Evaluate
    try:
        exp['Accuracy'] = accuracy_score(y, y_pred)
        exp['Recall'] = recall_score(y, y_pred, average='macro', zero_division=0)
        exp['F1'] = f1_score(y, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"ERROR calculating metrics: {e}")
        exp['Accuracy'] = 0
        exp['Recall'] = 0
        exp['F1'] = 0

    return exp



def evaluate_similarity(tests):
    file_path = './results/experiments.csv'
    experiments = load_experiments(file_path)
    for index, exp in experiments.iterrows():
        if exp['Technique'] == 'embedding' and exp['Method'] == 'similarity':
            domain = exp['Domain']
            print(exp)
            experiments.loc[index] = evaluate_similarity_experiment(exp, tests[domain])
    experiments.to_csv(file_path, index=False)


def evaluate_classifiers(tests):
    file_path = './results/experiments.csv'
    experiments = load_experiments(file_path)
    embeddings_base_path = './results/embeddings/'

    classifier_experiments = experiments[
        (experiments['Technique'] == 'embedding') & (experiments['Method'] == 'classifier')
    ].copy()

    classifier_experiments['model'] = classifier_experiments['Parameters'].apply(lambda p: p.get('model'))
    grouped_experiments = classifier_experiments.groupby('model')

    for model, group in grouped_experiments:

        print(f"--- Evaluating classifiers for model: {model} ---")
        embedding_col = f"{model}_embedding"
        embeddings_path = os.path.join(embeddings_base_path, f"{model}_embeddings.csv")
        instance_embeddings = pd.read_csv(embeddings_path)

        for index, exp in group.iterrows():
            domain = exp['Domain']
            print(f"Evaluating experiment ID: {exp['ID']}")
            test_df = tests[domain]
            merged_test_df = test_df.merge(
                instance_embeddings[['instance', embedding_col]],
                on='instance',
                how='left'
            )

            # Evaluate the experiment, modifying the original 'exp' Series is fine here
            # as the result is assigned back to the main DataFrame using .loc
            evaluated_exp = evaluate_classifier_experiment(exp, merged_test_df)

            # Update the main experiments DataFrame
            experiments.loc[index] = evaluated_exp

            # Save updated experiments
            experiments.to_csv(file_path, index=False)
        
