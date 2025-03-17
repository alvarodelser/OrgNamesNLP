from transformers import pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm



def avg_accuracy_score(y_true, y_pred):
    accuracy_list = []
    for label in y_pred.columns:  # For each label (column) in the DataFrame
        label_true = y_true[label.split("_",1)[1]]  # The class column is the predictions column minus the experiment id
        label_pred = y_pred[label]  # Predicted values for this label
        label_accuracy = accuracy_score(label_true, label_pred)
        accuracy_list.append(label_accuracy)
    return np.mean(accuracy_list)  # Average accuracy over all samples


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


def llm_classify(zero_shot_classifier, prompt: str, names: list, labels: list, multi_label: bool = False):
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