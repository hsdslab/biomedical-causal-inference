"""
Fine-tunes a Med-LLaMA3-8B model for sequence classification using LoRA quantization.
This script is intended to be run on a cluster with GPU access and handles data loading,
model training, calibration, and saving the results.
"""

import numpy as np
import torch
import pandas as pd
import os
import random
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import llama_train_src
import transformers

os.environ["WANDB_DISABLED"] = "true"
HF_TOKEN = "YOUR_HF_TOKEN_HERE"

# Speed + stability on A40
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_quantized_lora_model(model_id: str):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        quantization_config=bnb_config
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # typical for LLaMA
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)

    # Optional: print how many params are trainable (helps sanity-check)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return tokenizer, model


def full_process(split, dataset):
    set_seed(42)

    model_id = "YBXL/Med-LLaMA3-8B"
    tokenizer, model = build_quantized_lora_model(model_id)

    df_train, df_dev, df_test = llama_train_src.get_split(split, dataset)

    train_data = Dataset.from_pandas(df_train)
    test_data = Dataset.from_pandas(df_test)
    dev_data = Dataset.from_pandas(df_dev)

    def preprocess_function(examples):
        # Map label to int and keep it in the dataset
        result = tokenizer(
            examples["Temp_sentence"],
            truncation=True,
            max_length=480
   	     )
        result["labels"] = examples["label"]  # <-- add labels so trainer can compute loss
        return result

    tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
    tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)
    tokenized_dev = dev_data.map(preprocess_function, batched=True, remove_columns=dev_data.column_names)

    trainer = llama_train_src.train_transformer(tokenizer, model, tokenized_train, tokenized_dev)

    prob_dev, prob_test = llama_train_src.predict(trainer, tokenized_dev, tokenized_test)
    prob_test_cal = llama_train_src.calibration(prob_dev, df_dev["label"].tolist(), prob_test)

    df_res = pd.DataFrame(index=df_test.index)
    df_res["med_llama_temp"] = prob_test
    df_res["med_llama_temp_cal"] = prob_test_cal
    df_res["label"] = df_test["label"]

    # Define output directory relative to the repository root
    # e.g., ../../dat/<dataset>/proc/med_llama_temp
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(project_root, "dat", dataset, "proc", "med_llama_temp")
    os.makedirs(output_dir, exist_ok=True)
    df_res.to_csv(os.path.join(output_dir, f"df_res{split['dev'][0]}{split['test'][0]}.csv"))

    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    possible_splits = []
    for i in range(5):
        for j in range(5):
            if i != j:
                dicti = {"dev": [i], "test": [j], "train": [x for x in range(5) if x != i and x != j]}
                possible_splits.append(dicti)

    try:
        for split in possible_splits[16:]:
            full_process(split, "Analgesics-induced_acute_liver_failure")
            print(f"Completed fold: {split} ailf")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
