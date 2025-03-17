import capackage as ca

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from transformers import pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_train_test():
    df = ca.load_sparql_data()
    df = ca.clean_wd_data(df)
    # Plot the class distribution
    class_counts = df['category'].value_counts()
    print(class_counts)
    class_counts.plot(kind='barh', title='Distribution of Classes', figsize=(3, 0.5))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

    df = df.rename(columns={'category': 'label'})
    df['label'] = df['label'] == 'university hospital'
    return train_test_split(
        df['name'], df['label'], test_size=0.2, random_state=42
    )


evaluation_df = pd.DataFrame(columns=['method', 'model', 'parameters' , 'labels', 'predictions'])


def evaluate_reg_exp(train_test, evaluation_df = pd.DataFrame(columns=['method', 'model', 'parameters' , 'labels', 'predictions'])):
    X_train, X_test, y_train, y_test = train_test
    # MANUAL RULE-BASED CLASSIFIER
    predictions = ca.reg_exp_classifier(X_test)
    result_mr = pd.DataFrame({
        'method': 'Expression Matching',
        'model': 'N/A',
        'labels': y_test.tolist(),
        'predictions': predictions,
        'parameters': [{}] * len(y_test)
    })

    # AUTO RULE-BASED CLASSIFIER
    results_ar = []
    for tokens in range(1, 50, 1):
        _, predictions = ca.reg_exp_classifier_auto(X_train, y_train, X_test, tokens)
        results_ar.append(pd.DataFrame({
            'method': 'Expression Matching',
            'model': 'N/A',
            'labels': y_test.tolist(),
            'predictions': predictions,
            'parameters': [{'tokens': tokens}] * len(y_test)
        }))
    return pd.concat([evaluation_df] + [result_mr] + results_ar, ignore_index=True)

def evaluate_0shot(train_test, evaluation_df = pd.DataFrame(columns=['method', 'model', 'parameters' , 'labels', 'predictions'])):
    X_train, X_test, y_train, y_test = train_test
    # List of models for Zero-Shot Classification
    zero_shot_models = [
        ("facebook/bart-large-mnli", "BART"),
        ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "multilingual DeBERTa"),
        ("distilbert-base-uncased", "DistilBERT"),
        ("roberta-large-mnli", "RoBERTa"),
        ("xlnet-large-cased", "XLNet")
    ]
    results_0sb = []
    results_0st = []
    # Evaluate Zero-Shot Classifiers
    for model_id, model_name in zero_shot_models:
        print('Loading ' + model_name)
        classifier = pipeline("zero-shot-classification", model=model_id)
        # BINARY BASED CLASSIFIER
        predictions = ca.zero_shot_classifier_binary(X_test, classifier)
        results_0sb.append(pd.DataFrame({
            'method': '0-Shot',
            'model': model_name,
            'labels': y_test.tolist(),
            'predictions': predictions,
            'parameters': [{}] * len(y_test)
        })
        )
        # THRESHOLD BASED CLASSIFIER
        print('Loading ' + model_name + ' with threshold')
        scores = ca.zero_shot_classifier(X_test, classifier)
        for t in np.arange(0, 1.01, 0.01):
            predictions = scores > t
            results_0st.append(pd.DataFrame({
                'method': '0-Shot',
                'model': model_name,
                'labels': y_test.tolist(),
                'predictions': predictions,
                'parameters': [{'threshold': t}]* len(y_test)
            }))
    return pd.concat([evaluation_df] + results_0sb + results_0st, ignore_index=True)



def evaluate_semantic_sim(train_test, evaluation_df = pd.DataFrame(columns=['method', 'model', 'parameters' , 'labels', 'predictions'])):
    X_train, X_test, y_train, y_test = train_test
    # List of models for Semantic Similarity Classifier
    semantic_similarity_models = [
        ("paraphrase-MiniLM-L6-v2", "MiniLM"),
        ("sentence-transformers/all-MiniLM-L6-v2", "All MiniLM"),
        ("sentence-transformers/roberta-base-nli-stsb-mean-tokens", "RoBERTa"),
        ("sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens", "XLM-R")
    ]

    results_ssb = []
    results_sst = []
    # Evaluate Semantic Similarity Classifiers
    for model_id, model_name in semantic_similarity_models:
        print('Loading ' + model_name)
        model = SentenceTransformer(model_id)

        # BINARY BASED CLASSIFIER
        predictions = ca.semantic_similarity_classifier_binary(X_test, model)
        results_ssb.append(pd.DataFrame({
            'method': 'Semantic Similarity',
            'model': model_name,
            'labels': y_test.tolist(),
            'predictions': predictions,
            'parameters': [{}] * len(y_test)
        }))

        # THRESHOLD BASED CLASSIFIER
        print('Loading ' + model_name + ' with threshold')
        scores = ca.semantic_similarity_classifier(X_test, model)
        for t in np.arange(0, 1.01, 0.01):
            predictions = scores > t
            results_sst.append(pd.DataFrame({
                'method': 'Semantic Similarity',
                'model': model_name,
                'labels': y_test.tolist(),
                'predictions': predictions,
                'parameters': [{'threshold': t}] * len(y_test)
            }))
    return pd.concat([evaluation_df] + results_ssb + results_sst, ignore_index=True)


def evaluate_supervised(train_test, evaluation_df = pd.DataFrame(columns=['method', 'model', 'parameters' , 'labels', 'predictions'])):
    X_train, X_test, y_train, y_test = train_test
    # Define the models and their possible parameters
    models = {
        'Logistic Regression': (LogisticRegression, {'C': [1.0, 0.1, 0.01, 10, 100], 'solver': ['lbfgs', 'liblinear', 'newton-cg']}),
        'Support Vector Classifier': (SVC, {'C': [1.0, 0.1, 0.01, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}),
        'Random Forest': (RandomForestClassifier, {'n_estimators': [100, 200, 500], 'max_depth': [None, 5, 10, 20]})
    }

    results_tfidf = []
    results_w2v = []

    # Call the TF-IDF classifiers with different models and parameters
    for model_name, (model_class, params) in models.items():
        from itertools import product
        param_combinations = [dict(zip(params, v)) for v in product(*params.values())]
        for param_set in param_combinations:
            print(f'Loading {model_name} with params {param_set}')
            model = model_class(**param_set)
            _, tfidf_predictions = ca.tfidf_classifier(X_train, y_train, X_test, model)
            results_tfidf.append(pd.DataFrame({
                'method': 'TF-IDF',
                'model': model_name,
                'labels': y_test.tolist(),
                'predictions': tfidf_predictions,
                'parameters': [param_set] * len(y_test)
            }))

    # Call the Word2Vec classifiers with different models and parameters
    for model_name, (model_class, params) in models.items():
        from itertools import product
        param_combinations = [dict(zip(params, v)) for v in product(*params.values())]
        for param_set in param_combinations:
            print(f'Loading {model_name} with params {param_set}')
            model = model_class(**param_set)
            _, word2vec_predictions = ca.word2vec_classifier(X_train, y_train, X_test, model)
            results_w2v.append(pd.DataFrame({
                'method': 'Word2Vec',
                'model': model_name,
                'labels': y_test.tolist(),
                'predictions': word2vec_predictions,
                'parameters': [param_set] * len(y_test)
            }))

    return pd.concat([evaluation_df] + results_tfidf + results_w2v, ignore_index=True)

def compare_models(ax, df, method):
    # Color palettes
    model_count = df['model'].nunique()
    step = 1 / (model_count - 1)
    # Generate colors
    accuracy_colors = [sns.hls_palette(1, h=0.6, l=0.5, s=s)[0] for s in
                       [1 - step * i for i in range(model_count)]]  # Blueish
    recall_colors = [sns.hls_palette(1, h=0.3, l=0.5, s=s)[0] for s in
                     [1 - step * i for i in range(model_count)]]  # Greenish
    f1_colors = [sns.hls_palette(1, h=0.1, l=0.6, s=s)[0] for s in [1 - step * i for i in range(model_count)]]

    # Plot Baseline accuracy as a horizontal dashed line
    baseline_accuracy = df[(df['method'] == 'Majority Class')]['accuracy'].values[0]
    ax.axhline(y=baseline_accuracy, color='blue', linestyle='--',
               label=f'Baseline Accuracy ({baseline_accuracy:.2f})')

    # Filter data for the method
    rule_based_df = df[df['method'] == method]
    # Separate cases where 'parameters' is 'N/A' for special case plot
    special_case = rule_based_df[rule_based_df['parameters'] == {}].copy()
    normal_cases = rule_based_df[rule_based_df['parameters'] != {}].copy()

    # Extract float values from 'parameters' for the remaining cases
    normal_cases.loc[:, 'param_value'] = normal_cases['parameters'].str.extract(r'=(\d*\.?\d+)').astype(float)

    # Sort by the extracted parameter values to ensure smooth plotting
    normal_cases = normal_cases.sort_values(by='param_value')

    # Plot lines for accuracy, recall, and f1
    for idx, model in enumerate(normal_cases['model'].unique()):
        model_df = normal_cases[normal_cases['model'] == model]
        ax.plot(model_df['param_value'], model_df['accuracy'], color=accuracy_colors[idx], marker='.',
                label=f'Accuracy ({model})' if idx == 0 else "")
        ax.plot(model_df['param_value'], model_df['recall'], color=recall_colors[idx], marker='.',
                label=f'Recall ({model})' if idx == 0 else "")
        ax.plot(model_df['param_value'], model_df['f1'], color=f1_colors[idx], marker='.',
                label=f'F1 Score ({model})' if idx == 0 else "")

    # Plot special case as disconnected dots
    if not special_case.empty:
        x_min = normal_cases['param_value'].min()
        x_max = normal_cases['param_value'].max()
        x_shift = 0.1 * (x_max - x_min)
        for idx, model in enumerate(special_case['model'].unique()):
            model_df = special_case[special_case['model'] == model]
            ax.scatter(
                x=[x_min - x_shift * (1 + idx)] * len(model_df),  # Adjust x slightly to the left
                y=model_df['accuracy'],
                color=accuracy_colors[idx], marker='.', s=80, label=f'Accuracy ({model})' if idx == 0 else ""
            )
            ax.scatter(
                x=[x_min - x_shift * (1 + idx)] * len(model_df),
                y=model_df['recall'],
                color=recall_colors[idx], marker='.', s=80, label=f'Recall ({model})' if idx == 0 else ""
            )
            ax.scatter(
                x=[x_min - x_shift * (1 + idx)] * len(model_df),
                y=model_df['f1'],
                color=f1_colors[idx], marker='.', s=80, label=f'F1 Score ({model})' if idx == 0 else ""
            )

    # Add labels and title
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Value")
    ax.set_title("Metrics for " + method)


def get_metrics(train_test, evaluation_df):
    X_train, X_test, y_train, y_test = train_test
    results = []

    # Add baseline metrics
    majority_class = y_train.value_counts().idxmax()
    baseline_accuracy = accuracy_score(y_test, [majority_class] * len(y_test))
    baseline_recall = recall_score(y_test, [majority_class] * len(y_test), pos_label=True)
    baseline_f1 = f1_score(y_test, [majority_class] * len(y_test), pos_label=True)
    results.append({
        'method': 'Majority Class',
        'model': 'N/A',
        'parameters': {},
        'accuracy': baseline_accuracy,
        'recall': baseline_recall,
        'f1': baseline_f1
    })
    evaluation_df['parameters_str'] = evaluation_df['parameters'].apply(lambda x: json.dumps(x))
    for (method, model, parameters_str), group in evaluation_df.groupby(['method', 'model', 'parameters_str']):
        labels = np.array(group['labels'], dtype=int)
        predictions = np.array(group['predictions'], dtype=int)
        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, pos_label=True)
        f1 = f1_score(labels, predictions, pos_label=True)
        results.append({
            'method': method,
            'model': model,
            'parameters': json.loads(parameters_str) if parameters_str else {},
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1
        })
    evaluation_df.drop(columns=['parameters_str'], inplace=True)
    return pd.DataFrame(results)


def get_analytics(df):
    model_times = {
        '0-Shot':{
            'BART': 2.68,
            'multilingual DeBERTa': 3.67,
            'DistilBERT': 23.94,
            'RoBERTa': 3.45,
            'XLNet': 2.55
        },
        'Semantic Similarity' : {
            'MiniLM': 66.26,
            'All MiniLM': 65.20,
            'RoBERTa': 14.07,
            'XLM-R': 15.20
        }
    }
    # Calculate the unique methods and the maximum number of models per method
    methods = df['method'][df['method'] != 'Majority Class'].unique()
    col_count = df.groupby('method')['model'].nunique().max() + 1 # One extra column for best model comparison

    # Set up a grid of subplots with each method in a row and each model in a column
    fig, axs = plt.subplots(nrows=len(methods), ncols=col_count, figsize=(4 * col_count, 5 * len(methods)),
                            sharex=False, sharey=True)
    fig.suptitle("Metrics by Method and Model", fontsize=16)

    # Ensure axs is always a 2D array, even if there's only one row or column
    if len(methods) == 1:
        axs = [axs]  # Wrap in a list to keep 2D indexing for a single row
    if col_count == 1:
        axs = [[ax] for ax in axs]  # Wrap each ax in a list to keep 2D indexing for a single column

    def plot_method(axs, df, method):

        # Baseline accuracy line (assumes one baseline entry in data)
        baseline_accuracy = df[(df['method'] == 'Majority Class')]['accuracy'].values[0]

        # Filter data for the method
        method_df = df[df['method'] == method]
        models = method_df['model'].unique()


        if len(method_df['parameters'].iloc[0]) <= 1: #For one-dimensional parameter. The parameter length will be 1 or 0 for each row of the same method.
            special_case = method_df[method_df['parameters'].apply(lambda x: len(x) == 0)].copy() #We separate the binary decision as a special case.
            normal_cases = method_df[method_df['parameters'].apply(lambda x: len(x) == 1)].copy()

            key_name = list(normal_cases['parameters'].iloc[0].keys())[0]
            normal_cases['param_var'] = normal_cases['parameters'].apply(lambda x: x.get(key_name, None))
            normal_cases = normal_cases.sort_values(by='param_var')

            for col, model in enumerate(models):

                model_ax = axs[col]  # Get the subplot axis for the current model

                # Plot Baseline accuracy
                model_ax.axhline(y=baseline_accuracy, color='blue', linestyle='--',
                                 label=f'Baseline Accuracy ({baseline_accuracy:.2f})')

                model_df = normal_cases[normal_cases['model'] == model]
                # Plot lines for accuracy, recall, and F1
                model_ax.plot(model_df['param_var'], model_df['accuracy'], color='blue', marker='.',
                              label=f'Accuracy ({model})')
                model_ax.plot(model_df['param_var'], model_df['recall'], color='green', marker='.',
                              label=f'Recall ({model})')
                model_ax.plot(model_df['param_var'], model_df['f1'], color='red', marker='.',
                              label=f'F1 Score ({model})')
                # Plot special cases as disconnected dots if present
                special_model_df = special_case[special_case['model'] == model]
                x_min = normal_cases['param_var'].min()
                x_max = normal_cases['param_var'].max()
                x_shift = 0.1 * (x_max - x_min)
                if not special_model_df.empty:
                    for idx, row in special_model_df.iterrows():
                        x_min = x_min - x_shift
                        model_ax.scatter(x=x_min, y=row['accuracy'], color='blue', marker='o', s=80)
                        model_ax.scatter(x=x_min, y=row['recall'], color='green', marker='o', s=80)
                        model_ax.scatter(x=x_min, y=row['f1'], color='red', marker='o', s=80)
                # Labels and title for each model
                model_ax.set_title(model)
                model_ax.set_xlabel(key_name)
                model_ax.set_ylabel("Metric Value")
                model_ax.set_xlim(x_min - x_shift, x_max + x_shift)
                model_ax.set_ylim(0, 1)
                if col == 0:
                    axs[col].text(x_min-0.4*(x_max - x_min), 0.5, method, va='center', ha='left', rotation=90, fontsize=14)

            for col in range(len(models), col_count-1):
                axs[col].set_visible(False)  # Hide unused subplot

        elif len(method_df['parameters'].iloc[0]) == 2: #For one-dimensional parameter + categorical-parameter
            for col, model in enumerate(models):
                model_ax = axs[col]  # Get the subplot axis for the current model

                # Plot Baseline accuracy
                model_ax.axhline(y=baseline_accuracy, color='blue', linestyle='--',
                                 label=f'Baseline Accuracy ({baseline_accuracy:.2f})')

                model_df_filtered = method_df[method_df['model'] == model].copy()

                # Identify which parameter is continuous and which is categorical
                param_1, param_2 = model_df_filtered['parameters'].apply(lambda x: list(x.keys())).iloc[0]
                param_1_unique_values = model_df_filtered['parameters'].apply(lambda x: x.get(param_1, None)).nunique()
                param_2_unique_values = model_df_filtered['parameters'].apply(lambda x: x.get(param_2, None)).nunique()

                if param_1_unique_values > param_2_unique_values:  # param_1 is continuous
                    continuous_param = param_1
                    categorical_param = param_2
                else:  # param_2 is continuous
                    continuous_param = param_2
                    categorical_param = param_1

                # Create 'param_var' for continuous parameter
                model_df_filtered.loc[:, 'param_var'] = model_df_filtered['parameters'].apply(lambda x: x.get(continuous_param, None))
                model_df_filtered.loc[:, 'categorical_var'] = model_df_filtered['parameters'].apply(lambda x: x.get(categorical_param, None))
                model_df_filtered = model_df_filtered.sort_values(by='categorical_var')

                # Define distance between categorical values and the shift within each category
                categorical_shift = 6

                # Create the x-axis values based on the sorted order within each category
                model_df_filtered['x_axis'] = (
                        model_df_filtered.groupby('categorical_var').cumcount()-1  # Within-group order of param_var
                        + (model_df_filtered['categorical_var'].rank(method='dense').astype(int)-1) * categorical_shift
                # Shift by category
                )
                # Plot metrics for each model
                for i, (cat_val, cat_df) in enumerate(model_df_filtered.groupby('categorical_var')):
                    # Plot the metrics for each categorical value separately, using x_axis directly
                    model_ax.plot(cat_df['x_axis'], cat_df['accuracy'], color='blue', marker='.',
                                  label=f'Accuracy ({model})' if i == 0 else "")
                    model_ax.plot(cat_df['x_axis'], cat_df['recall'], color='green', marker='.',
                                  label=f'Recall ({model})' if i == 0 else "")
                    model_ax.plot(cat_df['x_axis'], cat_df['f1'], color='red', marker='.',
                                  label=f'F1 Score ({model})' if i == 0 else "")

                # Labels and title for each model
                model_ax.set_title(model)
                model_ax.set_xlabel(categorical_param, fontweight='bold', fontsize =10)
                model_ax.set_ylabel("Metric Value")

                # Set custom x-ticks to reflect categorical variable values
                cat_values = sorted(model_df_filtered['categorical_var'].unique())
                model_ax.set_xticks([i * categorical_shift for i in range(len(cat_values))])
                model_ax.set_xticklabels(cat_values)
                model_ax.tick_params(axis='x', which='major', pad=50)

                # Adjust the x and y limits
                model_ax.set_ylim(0, 1)
                x_min = model_df_filtered['x_axis'].min() - categorical_shift
                x_max =model_df_filtered['x_axis'].max() + categorical_shift
                model_ax.set_xlim(x_min, x_max)
                for idx, row in model_df_filtered.iterrows():
                    model_ax.text(row['x_axis'], -0.1, row['param_var'], ha='right', va='top', fontsize=9, color='black', rotation = 90)
                model_ax.text(x_min+categorical_shift*0.75, -0.1, continuous_param, ha='right', va='top', fontsize=10, fontweight='bold',
                                  color='black', rotation=45)
                if col == 0:
                    axs[col].text(x_min-0.25*(x_max - x_min), 0.5, method, va='center', ha='left', rotation=90, fontsize=14)

            for col in range(len(models), col_count-1):
                axs[col].set_visible(False)  # Hide unused subplot

        #Plot summary
        metrics_list = []
        labels = []
        for model in models:
            model_df = method_df[method_df['model'] == model]
            # Get best parameters
            best_row = model_df.loc[model_df['f1'].idxmax()]
            best_params = '\n'.join([f"{key}: {value}" for key, value in best_row['parameters'].items()])
            metrics_list.append([best_row['accuracy'], best_row['recall'], best_row['f1']])
            labels.append(model + '\n'+best_params)
        metrics_array = np.array(metrics_list)

        # Plot the grouped bar chart
        x = np.arange(len(models))  # The x positions for the models
        width = 0.2  # Width of each bar

        ax_s = axs[-1]  # Assuming axs is a 1D array of axes
        ax_s.bar(x - width, metrics_array[:, 0], width, label='Accuracy', color='blue')
        ax_s.bar(x, metrics_array[:, 1], width, label='Recall', color='green')
        ax_s.bar(x + width, metrics_array[:, 2], width, label='F1 Score', color='red')
        if method in ['0-Shot', 'Semantic Similarity']:
            ax_s_twin = axs.twinx()
            ax_s_twin.bar(x + 2*width, model_times.get(method,None).get(model,0), width, label='Iterations/s', color='gray')

        # Add labels and title
        ax_s.set_ylabel('Metric Value')
        ax_s.set_title(f'Best Models for {method}')
        ax_s.set_xticks(x)
        ax_s.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    preferred_order = ['Expression Matching', '0-Shot', 'Semantic Similarity', 'TF-IDF', 'Word2Vec']  # Define your desired order
    methods = [method for method in preferred_order if method in methods]
    # Plot each method's models on separate rows
    for row, method in enumerate(methods):
        plot_method(axs[row], df, method)

    # Adjust layout for better spacing
    plt.xticks(rotation=90)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reserve space for the title
    plt.show()