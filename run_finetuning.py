import sys
import os

# 1. Add the project root to sys.path so Python can find 'orgpackage'
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root)

# 2. Import functions from your modules
from orgpackage.aux import load_dataset
from orgpackage.finetuner import prepare_finetuning_data, train_val_split, train

# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------
OUTPUT_DIR = "results/finetuned_me5"
MODEL_NAME = "intfloat/multilingual-e5-base"
EPOCHS = 20
BATCH_SIZE = 64
LR = 2e-5
TEMPERATURE = 0.07
TRAIN_RATIO = 0.8
MAX_LENGTH = 128
PATIENCE = 3
SEED = 42
SAVE_EPOCHS = True
DRY_RUN = False


if __name__ == "__main__":
    print("[run_finetuning] Loading dataset...")
    df = load_dataset()
    
    print("[run_finetuning] Preparing data...")
    df = prepare_finetuning_data(df)
    
    if DRY_RUN:
        df = df.sample(min(500, len(df)), random_state=SEED).reset_index(drop=True)
        print(f"[run_finetuning] --dry_run: using {len(df)} rows.")

    print("[run_finetuning] Splitting into train/val subsets...")
    train_df, val_df = train_val_split(df, train_ratio=TRAIN_RATIO, seed=SEED)
    
    print("[run_finetuning] Starting training loop...")
    train(
        train_df=train_df,
        val_df=val_df,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        temperature=TEMPERATURE,
        max_length=MAX_LENGTH,
        patience=PATIENCE,
        seed=SEED,
        save_epochs=SAVE_EPOCHS
    )
