import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import gc

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




def compute_similarity(embeddings, class_embeddings):
    return (embeddings @ class_embeddings.T) * 100


def embedder(df, save_path="./results/embeddings.csv"):
    models_map = {
        'multilingual-e5': {'model_name': 'intfloat/multilingual-e5-large-instruct', 'max_length': 512, 'batch_size': 512},
        'qwen': {'model_name': 'Alibaba-NLP/gte-Qwen2-7B-instruct', 'max_length': 8192, 'batch_size': 32},
        'mistral': {'model_name': 'Linq-AI-Research/Linq-Embed-Mistral', 'max_length': 4096, 'batch_size': 256},
        'e5-small': {'model_name': 'intfloat/e5-small-v2', 'max_length': 512, 'batch_size': 1024},
    }
    if os.path.exists(save_path):
        df_saved = pd.read_csv(save_path)
    else:
        df_saved = df.copy()

    for model_key in models_map.keys():
        model_name = models_map[model_key]['model_name']
        max_length = models_map[model_key]['max_length']
        batch_size = models_map[model_key]['batch_size']
        embedding_column = model_key + '_embedding'

        if embedding_column not in df_saved.columns:
            df_saved[embedding_column] = [None] * len(df_saved)
        mask = df_saved[embedding_column].isna()
        start_idx = mask.idxmax() if mask.any() else len(df_saved)
        if start_idx >= len(df_saved):
            print(f"All embeddings for {model_key} already exist.")
            continue

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
            if save_path and (i - start_idx) // batch_size % 50 == 0:
                df_saved.to_csv(save_path, index=False)

        df_saved.to_csv(save_path, index=False)
        print(f"Final embeddings saved for {model_key}")

        # Freeing memory
        del model
        del tokenizer
        torch.mps.empty_cache()  # Free MPS memory
        gc.collect()
    return df_saved