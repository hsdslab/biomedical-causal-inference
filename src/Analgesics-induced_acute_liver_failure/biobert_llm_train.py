"""
Training script for BioBERT model specifically utilizing LLM-generated sentences.
This script performs cross-validation, trains the transformer, applies Isotonic Regression for calibration,
and saves the output predictions.
"""
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
import pandas as pd
import os
from transformers import DataCollatorWithPadding, AdamW, EarlyStoppingCallback, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
import random
import biobert_llm_train_src
from datasets import Dataset
import shutil
from tqdm import tqdm

import os
os.environ["WANDB_DISABLED"] = "true"

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", num_labels=2, ignore_mismatched_sizes=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def full_process(split, dataset, model, tokenizer):

    # Set a global seed
    set_seed(42)

    df_train, df_dev, df_test = biobert_llm_train_src.get_split(split, dataset)

    train_data = Dataset.from_pandas(df_train)
    test_data = Dataset.from_pandas(df_test)
    dev_data = Dataset.from_pandas(df_dev)

    def preprocess_function(examples):
        # Lowercase the text separately
        examples["llm_sentence"] = [sentence.lower() for sentence in examples["llm_sentence"]]
        return tokenizer(examples["llm_sentence"], truncation=True, max_length=128)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)
    tokenized_dev = dev_data.map(preprocess_function, batched=True)

    trainer = biobert_llm_train_src.train_transformer(tokenizer, model, tokenized_train, tokenized_dev)

    prob_dev, prob_test = biobert_llm_train_src.predict(trainer, tokenized_dev, tokenized_test)

    prob_test_cal = biobert_llm_train_src.calibration(prob_dev, df_dev["label"].tolist(), prob_test)

     # delete results and logs dir
    shutil.rmtree("results")
    shutil.rmtree("logs")
    #shutil.rmtree("mlruns")
    #shutil.rmtree("wandb")

    df_res = pd.DataFrame()
    df_res.index = df_test.index
    df_res["biobert_llm"] = prob_test
    df_res["biobert_llm_cal"] = prob_test_cal
    df_res["label"] = df_test["label"]

    # Ensure relative paths for repository portability
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "dat", dataset, "proc", "cross_val_biobert_llm")
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, f'df_res{split["dev"][0]}{split["test"][0]}.csv'))
    
    return None

possible_splits = []

        
for i in range(5):
    for j in range(5):
        if i !=j:
            dicti = {"dev": [i], "test": [j], "train": [x for x in range(5) if x != i and x != j]}
            possible_splits.append(dicti)
        
for split in tqdm(possible_splits):
    full_process(split, "Analgesics-induced_acute_liver_failure", model, tokenizer)