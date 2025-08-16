# train_reviewer_final_patched.py

import pandas as pd
import re
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig  # <-- Make sure this is imported
)
import numpy as np
import evaluate
import sys

print("--- Environment and Libraries Ready ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# --- Data Loading and Preparation ---
def is_python_code(code_string: str) -> bool:
    if not isinstance(code_string, str) or not code_string.strip(): return False
    python_indicators = [
        r'\bdef\s+\w+\(.*\):', r'\bclass\s+\w+:', r'\bimport\s+[\w\.]+',
        r'#\s*-\*-\s*coding:\s*utf-8\s*-\*-', r'if __name__ == "__main__":'
    ]
    lines = code_string.split('\n')
    if not lines: return False
    indented_lines = sum(1 for line in lines if line.startswith('    ') and line.strip())
    if len(lines) > 0 and (indented_lines / len(lines) > 0.1): return True
    for indicator in python_indicators:
        if re.search(indicator, code_string, re.IGNORECASE): return True
    return False

try:
    df_train = pd.read_json('./drive/MyDrive/msg-train.jsonl', lines=True)
    df_valid = pd.read_json('msg_valid_python.jsonl', lines=True)
    df_test = pd.read_json('msg_test_python.jsonl', lines=True)
    print("Successfully loaded dataset files.")
except FileNotFoundError as e:
    print(f"!!! ERROR: Dataset file not found: {e}. Please update your file paths. !!!")
    exit()

df_train['is_python'] = df_train['oldf'].apply(is_python_code)
df_train_py = df_train[df_train['is_python']].copy()
df_valid_py = df_valid[df_valid['lang'] == 'py'].copy()
df_test_py = df_test[df_test['lang'] == 'py'].copy()

def format_diff_for_model(patch):
    if not isinstance(patch, str): return ""
    formatted_lines = []
    for line in patch.split('\n'):
        if line.startswith('+ '): formatted_lines.append(f"[ADD] {line[1:].strip()}")
        elif line.startswith('- '): formatted_lines.append(f"[DEL] {line[1:].strip()}")
        elif line.strip() and not line.startswith('@@'): formatted_lines.append(f"[KEEP] {line.strip()}")
    return " ".join(formatted_lines)

for df in [df_train_py, df_valid_py, df_test_py]:
    df['model_input'] = df['patch'].apply(format_diff_for_model)
    df.dropna(subset=['msg', 'model_input'], inplace=True)
    df = df[df['model_input'].str.len() > 0].copy()


# # Take a small sample for a quick test run
# df_train_py_sample = df_train_py.sample(n=1000, random_state=42)
# df_valid_py_sample = df_valid_py.sample(n=100, random_state=42)

# # --- Then, when creating the DatasetDict, use these smaller dataframes ---
# all_datasets = DatasetDict({
#     'train': Dataset.from_pandas(df_train_py_sample[['model_input', 'msg']]),
#     'validation': Dataset.from_pandas(df_valid_py_sample[['model_input', 'msg']]),
#     # You can keep the test set as is
#     'test': Dataset.from_pandas(df_test_py[['model_input', 'msg']])
# })


all_datasets = DatasetDict({
    'train': Dataset.from_pandas(df_train_py[['model_input', 'msg']]),
    'validation': Dataset.from_pandas(df_valid_py[['model_input', 'msg']]),
    'test': Dataset.from_pandas(df_test_py[['model_input', 'msg']])
})

# --- Model Configuration and Tokenization ---
MODEL_CHECKPOINT = "Salesforce/codet5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
tokenizer.add_special_tokens({'additional_special_tokens': ['[ADD]', '[DEL]', '[KEEP]']})

def tokenize_function(examples):
    inputs = tokenizer(examples["model_input"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["msg"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_datasets = all_datasets.map(tokenize_function, batched=True, remove_columns=['model_input', 'msg'])

# --- Model Training ---
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
model.resize_token_embeddings(len(tokenizer))

output_dir = "codet5-finetuned-v3"

training_args = TrainingArguments(
    output_dir=output_dir,
    # evaluation_strategy="epoch",
    do_train=True,                    # Explicitly tell the trainer to train
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3, # <-- Change this to 5
    lr_scheduler_type="linear", # <-- ADD THIS LINE
    warmup_steps=500,
    # predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none",
)

# ❗❗❗ CODE PATCH ❗❗❗
# Manually create the 'generation_config' that the trainer is missing.
# This directly solves the AttributeError.
training_args.generation_config = GenerationConfig.from_model_config(model.config)


bleu_metric = evaluate.load("bleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("--- Starting Model Fine-Tuning ---")
trainer.train()
print("--- Model Training Complete ---")

final_model_path = f"./{output_dir}/final_model"
trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")

 