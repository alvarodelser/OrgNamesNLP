import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

from orgpackage.config import STRUCTURE_MAPPING, DOMAIN_CLASSES_CORR, EMB_MODELS
from orgpackage.aux import load_experiments, get_id, prepare_labels
from orgpackage.ruleclassifier import country_word_generator
from orgpackage.evaluator import evaluate_rule_experiment, evaluate_similarity_experiment, evaluate_classifier_experiment




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
            exp = evaluate_similarity_experiment(exp, validation)
        validation_exps.loc[index] = exp
    
    best_exp = validation_exps.loc[validation_exps['F1'].idxmax()]
    best_exp_df = pd.DataFrame([best_exp], columns=best_exp.index)

    # Saving optimization results
    param_exps = pd.concat([param_exps, validation_exps], ignore_index=True)
    param_exps.to_csv(file_path, index=False)
    return best_exp_df


############################################################# RULES ######################################################
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
                        val_new_row = pd.DataFrame([[id, domain, technique, method, params, None, None, None]],
                                           columns=["ID", "Domain", "Technique", "Method", "Parameters", "Accuracy", "Recall", "F1"])
                        validation_exps = pd.concat([validation_exps, val_new_row], ignore_index=True)

                    new_row = optimize_parameter(validation_exps, validation, 'token_num')
                    experiments = pd.concat([experiments, new_row], ignore_index=True)

    experiments.to_csv(file_path, index=False)


############################################################# NLI ######################################################

def get_prompts(train, classes):
    one_shot_examples = []
    few_shot_examples = []
    for cls in classes:
        class_df = train[train[cls] == 1]
        if not class_df.empty:
            # Sample once per class for one-shot
            sample = class_df.sample(n=1, random_state=42)['names'].iloc[0]
            one_shot_examples.append(f"'{sample}' -> {cls}")

            # Sample once per class and country for few-shot
            for country in class_df['country'].unique():
                country_df = class_df[class_df['country'] == country]
                if not country_df.empty:
                    sample = country_df.sample(n=1, random_state=42)['names'].iloc[0]
                    few_shot_examples.append(f"'{sample}' -> {cls}")

    # Now sample others class
    others_df = train[(train[classes] == 0).all(axis=1)]
    sample = others_df.sample(n=1, random_state=42)['names'].iloc[0]
    one_shot_examples.append(f"'{sample}' -> others")
    for country in others_df['country'].unique():
        country_df = others_df[others_df['country'] == country]
        if not country_df.empty:
            sample = country_df.sample(n=1, random_state=42)['names'].iloc[0]
            few_shot_examples.append(f"'{sample}' -> others")

    few_shot_sample = "\n".join(few_shot_examples)
    one_shot_sample = "\n".join(one_shot_examples)
    prompts = {
        "0_shot": "This organization is a {}",
        "1_shot": f"""Context: Here are some classified examples of organizations:\n{one_shot_sample}\nTask: This organization is a {{}}.""",
        "few_shot": f"""Context: Here are some classified examples of organizations in different countries:\n{few_shot_sample}\nTasks: This organization is a {{}}."""
    }
    return prompts



############################################################# EMBEDDIGS ######################################################
def train_classifier(train, exp):
    import ast
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.utils.class_weight import compute_class_weight

    domain = exp['Domain']
    classifier_type = exp['Parameters']['classifier']
    structure = exp['Parameters']['structure']
    model = exp['Parameters']['model']
    classes = DOMAIN_CLASSES_CORR[domain]
    embedding_col = model + '_embedding'


    # Parse embeddings
    X, valid_indices = [], []
    for idx, emb in enumerate(train[embedding_col]):
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
        raise ValueError(f"No valid embeddings found for domain {domain}")

    X = np.array(X)
    filtered_train = train.iloc[valid_indices].copy()
    y = prepare_labels(filtered_train, classes, structure)

    # Helper: create classifier with correct params
    def make_base_classifier(weight=None):
        if classifier_type == 'logreg':
            return LogisticRegression(
                C=exp['Parameters']['C'],
                solver=exp['Parameters']['solver'],
                penalty=exp['Parameters']['penalty'],
                max_iter=1000,
                random_state=42,
                class_weight=weight or 'balanced'
            )
        elif classifier_type == 'svm':
            return SVC(
                C=exp['Parameters']['C'],
                kernel=exp['Parameters']['kernel'],
                gamma=exp['Parameters'].get('gamma', 'scale'),
                probability=True,
                random_state=42,
                class_weight=weight or 'balanced'
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Compute class weights if possible
    if structure == '2-class':
        try:
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            class_weight = dict(zip(np.unique(y), weights))
        except:
            class_weight = 'balanced'
    else:
        class_weight = 'balanced'  # for 3-multiclass and nested-class (both multilabel)

    # Handle nested structure (only for medical)
    if structure == 'nested-class':
        clf_dict = {}
        hospital_idx = classes.index('hospital')
        y_hospital = y[:, hospital_idx]
        clf_hosp = make_base_classifier()
        clf_hosp.fit(X, y_hospital)
        clf_dict['hospital'] = clf_hosp

        mask = filtered_train['hospital'] == 1
        X_hosp = X[mask.to_numpy()]
        y_univ = filtered_train.loc[mask, 'university_hospital'].values
        if len(X_hosp) > 0:
            clf_univ = make_base_classifier()
            clf_univ.fit(X_hosp, y_univ)
            clf_dict['university_hospital'] = clf_univ

        return clf_dict

    # Handle nested structure (only for medical)
    base_clf = make_base_classifier(class_weight)
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X, y)
    return clf

