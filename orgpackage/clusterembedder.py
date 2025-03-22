import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel

import fasttext
import fasttext.util

import gc

from orgpackage.config import EMB_MODELS


def get_all_languages():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data/country_dictionary.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        COUNTRIES_DICT = json.load(f)
    ls = set()
    for country in COUNTRIES_DICT:
        language = COUNTRIES_DICT.get(country, {}).get('languages', [None])[0]
        ls.add(language)
    return ls

def get_downloaded_languages():
    project_root = Path(__file__).resolve().parent.parent
    dir_path = project_root / "fasttext_models"
    pattern = re.compile(r"cc\.(\w{2,3})\.300\.bin")
    dls = set()
    for filename in os.listdir(dir_path):
        match = pattern.match(filename)
        if match:
            dls.add(match.group(1))
    return dls


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def fasttext_mean_embedding(model, text):
    words = text.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model]
    if not word_vectors:
        return [0] * model.get_dimension()  # Return zero vector if no valid words
    return list(sum(word_vectors) / len(word_vectors))  # Compute mean embedding


def embedder(df, model_key):
    save_path = f"./results/embeddings/{model_key}_embeddings.csv"
    embedding_column = model_key + '_embedding'
    if os.path.exists(save_path):
        df_saved = pd.read_csv(save_path)
    else:
        df_saved = df.copy()

    # FAST TEXT MEAN WORD EMBEDDING - I NEED TO SPLIT INTO LANGUAGES AGAIN
    if model_key == 'fasttext':
        with open('./data/country_dictionary.json', 'r', encoding='utf-8') as f:
            COUNTRIES_DICT = json.load(f)

        if os.path.exists(save_path):
            df_saved = pd.read_csv(save_path)
            processed_instances = set(df_saved["instance"])
        else:
            df_saved = pd.DataFrame(columns=["instance", "language", "names", embedding_column])
            processed_instances = set()

        df['language'] = df['country'].apply(lambda x: COUNTRIES_DICT[x]['languages'][0])
        print(df)
        df_to_process = df[~df["instance"].isin(processed_instances)]
        ls = df_to_process["language"].unique().tolist()
        dls = get_downloaded_languages()
        print(f'languages to review: {dls}')

        for lang in ls:
            if lang not in dls:
                # Skip if you do not have the model downloaded yet
                print(f"Model not found for language '{lang}' – skipping for now.")
                continue
            lang_df = df_to_process[df_to_process["language"] == lang]
            model_path = f'./fasttext_models/cc.{lang}.300.bin'
            print(f"Loading model: {model_path}")
            ft_model = fasttext.load_model(model_path)
            for i, row in tqdm(lang_df.iterrows(), total=len(lang_df), desc=f"Embedding lang={lang}"):
                instance_id = row["instance"]
                language = row["language"]
                name = row["names"]
                if isinstance(name, list):
                    name = name[0]

                # Compute the embedding
                try:
                    embedding = fasttext_mean_embedding(ft_model, str(name))
                except Exception as e:
                    print(f"Error processing instance {instance_id}: {e}")
                    embedding = None
                df_saved.loc[len(df_saved)] = [instance_id, name, embedding]
                processed_instances.add(instance_id)
            df_saved.to_csv(save_path, index=False)
            del ft_model
            gc.collect()
    else:
        # ENCODER-BASED EMBEDDINGS
        model_name = EMB_MODELS[model_key]['model_name']
        max_length = EMB_MODELS[model_key]['max_length']
        batch_size = EMB_MODELS[model_key]['batch_size']

        if embedding_column not in df_saved.columns:
            df_saved[embedding_column] = [None] * len(df_saved)
        mask = df_saved[embedding_column].isna()
        start_idx = mask.idxmax() if mask.any() else len(df_saved)
        if start_idx >= len(df_saved):
            print(f"All embeddings for {model_key} already exist.")
            return

        print(f"Loading model: {model_key}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        print(f"Generating embeddings for {model_key} starting from index {start_idx}")
        names = df_saved['names'].tolist()

        for i in tqdm(range(start_idx, len(names), batch_size), desc=f"Generating embeddings for {model_key}"):
            batch_names = names[i:i + batch_size]  # Batch processing
            batch_dict = tokenizer(batch_names, max_length=max_length, padding=True, truncation=True,
                                   return_tensors='pt')
            # Calculate batch embeddings
            with torch.no_grad():  # Disable gradient computation
                outputs = model(**batch_dict)

            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Update df per batch
            df_saved.loc[i:i + len(embeddings) - 1, embedding_column] = pd.Series(embeddings.tolist(), index=range(i, i + len(embeddings)))


            #Save every 50 batches
            if save_path and (i - start_idx) // batch_size % 10 == 0:
                df_saved.to_csv(save_path, index=False)

        df_saved.to_csv(save_path, index=False)
        print(f"Final embeddings saved for {model_key}")

        # Freeing memory
        del model
        del tokenizer
        torch.mps.empty_cache()  # Free MPS memory
        gc.collect()