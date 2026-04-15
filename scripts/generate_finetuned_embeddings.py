import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import argparse
from orgpackage.aux import load_dataset

def mean_pool_normalize(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool last hidden state over non-padding tokens, then L2-normalise."""
    last_hidden = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return F.normalize(pooled, p=2, dim=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="results/finetuned_me5")
    parser.add_argument("--output_path", type=str, default="results/embeddings/finetuned-me5_embeddings.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    df = load_dataset()
    if args.limit:
        df = df.head(args.limit)
    
    print(f"Loading model from {args.model_path}...")
    # Using trust_remote_code=True if needed, though for mE5 it usually isn't
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)
    model.eval()

    names = df['names'].tolist()
    # Add "query: " prefix as required by mE5 for retrieval/classification tasks
    # We use "query: " because these are the strings we want to classify
    prefixed_names = [f"query: {name}" for name in names]
    
    all_embeddings = []
    
    print(f"Generating embeddings for {len(prefixed_names)} instances...")
    with torch.no_grad():
        for i in range(0, len(prefixed_names), args.batch_size):
            batch_names = prefixed_names[i : i + args.batch_size]
            inputs = tokenizer(batch_names, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = mean_pool_normalize(outputs, inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu().numpy())
            
            if (i // args.batch_size) % 10 == 0:
                 print(f"  Processed {i}/{len(prefixed_names)}...")

    all_embeddings = np.vstack(all_embeddings)
    
    # Save results
    # We only need instance, names, and the embedding for the standard format
    output_df = df[['instance', 'names']].copy()
    output_df['finetuned-me5_embedding'] = [list(emb) for emb in all_embeddings]
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_df.to_csv(args.output_path, index=False)
    print(f"Saved embeddings to {args.output_path}")

    # Update label embeddings
    labels_csv = 'results/embeddings/label_embeddings.csv'
    if os.path.exists(labels_csv):
        print("Updating label embeddings...")
        labels_df = pd.read_csv(labels_csv)
        labels = labels_df['label'].tolist()
        # Labels are also queries in this context (asymmetric comparison)
        prefixed_labels = [f"query: {label}" for label in labels]
        
        with torch.no_grad():
            inputs = tokenizer(prefixed_labels, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            label_embeddings = mean_pool_normalize(outputs, inputs['attention_mask'])
            
        labels_df['finetuned-me5_embedding'] = [list(emb) for emb in label_embeddings.cpu().numpy()]
        labels_df.to_csv(labels_csv, index=False)
        print(f"Updated {labels_csv}")

if __name__ == "__main__":
    main()
