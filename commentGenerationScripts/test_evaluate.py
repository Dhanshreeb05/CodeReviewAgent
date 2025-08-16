# test_model_v3_final.py

import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


def format_diff_for_model(patch: str) -> str:
    if not isinstance(patch, str): return ""
    formatted_lines = []
    for line in patch.split('\n'):
        if line.startswith('+ '): formatted_lines.append(f"[ADD] {line[1:].strip()}")
        elif line.startswith('- '): formatted_lines.append(f"[DEL] {line[1:].strip()}")
        elif line.strip() and not line.startswith('@@'): formatted_lines.append(f"[KEEP] {line.strip()}")
    return " ".join(formatted_lines)


# ==============================================================================
# Main Evaluation Logic
# ==============================================================================
TOKENIZER_CHECKPOINT = "Salesforce/codet5-base"
LOCAL_MODEL_PATH = "./codet5-finetuned-v3/final_model"

print("--- Loading Model and Tokenizer ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)
    bleu_metric = evaluate.load("bleu")
    if torch.cuda.is_available():
        model.to("cuda")
        print("✅ Model moved to GPU.")
    print("✅ Model, tokenizer, and BLEU metric loaded successfully!")
except OSError:
    print(f"❌ Error: Model not found at '{LOCAL_MODEL_PATH}'.")
    exit()

try:
    df_test = pd.read_json('msg_test_python.jsonl', lines=True)
    print(f"✅ Loaded {len(df_test)} examples from the test set.")
except FileNotFoundError:
    print("❌ Error: 'msg_test_python.jsonl' not found.")
    exit()

actual_comments = []
generated_comments = []

print("\n--- Generating predictions for the entire test set... ---")
for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
    patch = row['patch']
    actual_comment = row['msg']
    formatted_diff = format_diff_for_model(patch)

    # ❗❗❗ THE FIX: Make this tokenizer call identical to the training one ❗❗❗
    # This explicitly truncates long inputs to the model's maximum context size.
    inputs = tokenizer(
        formatted_diff,
        return_tensors="pt",
        max_length=512,        # Enforce max length
        truncation=True,       # Enable truncation
        padding="max_length"   # Pad shorter sequences
    ).input_ids

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    outputs = model.generate(
        inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    generated_comment = tokenizer.decode(outputs[0], skip_special_tokens=True)

    actual_comments.append([actual_comment])
    generated_comments.append(generated_comment)

print("\n--- Calculating Final BLEU Scores ---")
results = bleu_metric.compute(predictions=generated_comments, references=actual_comments)
precisions = results['precisions']

print("\n" + "#"*50)
print("### FINAL MODEL EVALUATION REPORT ###")
print("#"*50)
print(f"Total Test Examples: {len(generated_comments)}")
print(f"Overall BLEU Score:  {results['bleu']:.4f}")
print("---")
print(f"BLEU-1 (Individual Words): {precisions[0]:.4f}")
print(f"BLEU-2 (Two-word Phrases): {precisions[1]:.4f}")
print(f"BLEU-3 (Three-word Phrases): {precisions[2]:.4f}")
print(f"BLEU-4 (Four-word Phrases):  {precisions[3]:.4f}")
print("#"*50)