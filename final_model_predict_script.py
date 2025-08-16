import torch
import torch.nn as nn
from transformers import AutoModel
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn as nn

class CodeReviewClassifier(nn.Module):
    """
    Our neural network model that combines:
    1. CodeBERT (understands code) for processing the patch text
    2. A small network for numerical features  
    3. A classifier that combines both to make the final prediction
    """
    
    def __init__(self, model_name='microsoft/codebert-base', num_numerical_features=8):
        super().__init__()
        
        # Load pre-trained CodeBERT (knows about code already)
        self.codebert = AutoModel.from_pretrained(model_name)
        self.codebert_dim = self.codebert.config.hidden_size  # Usually 768
        
        # Small network to process numerical features
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Final classifier that combines everything
        combined_dim = self.codebert_dim + 32  # 768 + 32 = 800
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # 2 classes: needs review or not
        )
        
    def forward(self, input_ids, attention_mask, numerical_features):
        # Process the code patch with CodeBERT
        codebert_output = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the [CLS] token representation (summary of the whole sequence)
        code_features = codebert_output.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]
        
        # Process numerical features
        numerical_processed = self.numerical_processor(numerical_features)  # Shape: [batch_size, 32]
        
        # Combine both types of features
        combined_features = torch.cat([code_features, numerical_processed], dim=1)  # Shape: [batch_size, 800]
        
        # Make final prediction
        logits = self.classifier(combined_features)  # Shape: [batch_size, 2]
        
        return logits


def load_trained_model(model_dir="response/model_85.67"):
    """
    Load a previously trained model for making predictions
    """
    import json
    from transformers import AutoTokenizer
    
    print(f"Loading model from {model_dir}...")
    
    # Load model configuration
    with open(f"{model_dir}/model_config.json", 'r') as f:
        config = json.load(f)
    
    # Load tokenizer with fallback (same fix as your services)
    try:
        # Try loading saved tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
        print("✓ Loaded saved tokenizer")
    except Exception as e:
        print(f"Failed to load saved tokenizer: {e}")
        print("Rebuilding tokenizer from original...")
        
        # Fallback: Load original and add special tokens
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        special_tokens = ['[ADD]', '[DEL]', '[KEEP]', '[SEP]']
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        print("✓ Rebuilt tokenizer successfully")
    
    print(f"✓ Tokenizer ready with {len(tokenizer)} tokens")
    
    # Create model with same architecture
    model = CodeReviewClassifier(
        model_name=config['model_name'],
        num_numerical_features=config['num_numerical_features']
    )
    
    # IMPORTANT: Resize embeddings to match the tokenizer
    model.codebert.resize_token_embeddings(len(tokenizer))
    
    # Load the trained weights
    model.load_state_dict(torch.load(f"{model_dir}/model_weights.pt", map_location='cpu'))
    model.eval()  # Set to evaluation mode
    print(f"✓ Model loaded successfully")
    
    return model, tokenizer, config

def preprocess_single_patch(patch_text, msg_text="", lang="undefined", proj="unknown"):
    """
    Preprocess a single code patch for prediction
    """
    # Create a simple preprocessor (same logic as training)
    def clean_patch_for_inference(patch):
        if not patch or patch.strip() == '':
            return ''
        
        # Clean up whitespace
        patch = re.sub(r'\s+', ' ', patch)
        
        # Process each line in the diff
        lines = patch.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            elif line.startswith('@@'):
                processed_lines.append(f"[SEP] {line}")
            elif line.startswith('+'):
                content = line[1:].strip()
                processed_lines.append(f"[ADD] {content}")
            elif line.startswith('-'):
                content = line[1:].strip()
                processed_lines.append(f"[DEL] {content}")
            else:
                processed_lines.append(f"[KEEP] {line}")
        
        return ' '.join(processed_lines)
    
    # Process the patch
    cleaned_patch = clean_patch_for_inference(patch_text)
    
    # Extract numerical features
    patch_length = len(patch_text)
    num_additions = patch_text.count('+') - patch_text.count('@@')  # Exclude @@ lines
    num_deletions = patch_text.count('-') - patch_text.count('@@')
    total_changes = max(0, num_additions) + max(0, num_deletions)
    
    has_message = 1 if msg_text and msg_text.strip() != '' else 0
    message_length = len(msg_text) if msg_text else 0
    
    is_python = 1 if lang == 'py' else 0
    is_undefined_lang = 1 if lang == 'undefined' else 0
    
    # Create feature dictionary
    features = {
        'processed_patch': cleaned_patch,
        'patch_length': patch_length,
        'num_additions': max(0, num_additions),
        'num_deletions': max(0, num_deletions), 
        'total_changes': total_changes,
        'has_message': has_message,
        'message_length': message_length,
        'is_python': is_python,
        'is_undefined_lang': is_undefined_lang
    }
    
    return features

def predict_code_review(model, tokenizer, patch_text, msg_text="", lang="undefined", proj="unknown"):
    """
    Predict whether a code patch needs review
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        patch_text: Git diff text
        msg_text: Optional message/comment
        lang: Programming language
        proj: Project name
    
    Returns:
        Dictionary with prediction and confidence
    """
    device = next(model.parameters()).device
    
    # Preprocess the input
    features = preprocess_single_patch(patch_text, msg_text, lang, proj)
    
    # Tokenize the patch
    encoding = tokenizer(
        features['processed_patch'],
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    # Prepare numerical features
    numerical_features = torch.tensor([
        features['patch_length'],
        features['num_additions'],
        features['num_deletions'],
        features['total_changes'],
        features['has_message'],
        features['message_length'],
        features['is_python'],
        features['is_undefined_lang']
    ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    numerical_features = numerical_features.to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask, numerical_features)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Format results
    class_names = ['No Review Needed', 'Review Needed']
    result = {
        'prediction': predicted_class,
        'prediction_label': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'no_review_needed': probabilities[0][0].item(),
            'review_needed': probabilities[0][1].item()
        },
        'features_used': features
    }
    
    return result

# ==============================================================================
# 1. PREPROCESSING FUNCTION
# This must be the same function used during training.
# ==============================================================================
def format_diff_for_model(patch: str) -> str:
    """
    Formats a git diff by prepending special tokens ([ADD], [DEL], [KEEP])
    to each line for the model to understand the change type.
    """
    if not isinstance(patch, str):
        return ""
    formatted_lines = []
    for line in patch.split('\n'):
        if line.startswith('+ '):
            formatted_lines.append(f"[ADD] {line[1:].strip()}")
        elif line.startswith('- '):
            formatted_lines.append(f"[DEL] {line[1:].strip()}")
        elif line.strip() and not line.startswith('@@'):
            formatted_lines.append(f"[KEEP] {line.strip()}")
    return " ".join(formatted_lines)


# ==============================================================================
# 2. MODEL LOADING
# ==============================================================================
# Define the checkpoints for the tokenizer and your local model
# Note: Use the tokenizer that matches the model you want to use (e.g., codet5-large)
TOKENIZER_CHECKPOINT = "Salesforce/codet5-large"

# !!!IMPORTANT: Update this path to your best-performing model folder for Comment Generation
COMMENT_GENERATION_MODEL_PATH = "commentGeneration/models"

print(f"--- Loading model from: {COMMENT_GENERATION_MODEL_PATH} ---")

try:
    # Load the tokenizer from the Hub to avoid corruption issues
    tokenizer_comment = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)

    # Load your fine-tuned model from the local directory
    model_comment = AutoModelForSeq2SeqLM.from_pretrained(COMMENT_GENERATION_MODEL_PATH)

    # Move model to GPU if available for faster generation
    if torch.cuda.is_available():
        model_comment.to("cuda")
        print("✅ Model moved to GPU.")

    print("✅ Model and tokenizer loaded successfully!")
except OSError:
    print(f"❌ Error: Model not found at '{COMMENT_GENERATION_MODEL_PATH}'.")
    print("Please make sure the path is correct.")
    exit()


# ==============================================================================
# 3. COMMENT GENERATION
# ==============================================================================
def generate_review_comment(diff_text: str):
    """
    Takes a raw diff string and generates a review comment.
    """
    # Preprocess the diff text
    formatted_diff = format_diff_for_model(diff_text)

    # Tokenize the input, ensuring it's truncated correctly
    inputs = tokenizer_comment(
        formatted_diff,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).input_ids

    # Move tensor to the same device as the model
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # Generate the comment using the model
    outputs = model_comment.generate(
        inputs,
        max_length=128,  # Max length of the generated comment
        num_beams=5,  # Use beam search for higher quality results
        early_stopping=True
    )

    # Decode the output tokens into a human-readable string
    comment = tokenizer_comment.decode(outputs[0], skip_special_tokens=True)
    return comment


# Load trained model
# !!!CHANGE THIS PATH TO MODEL DIRECTORY OF CODE REVIEWER MODEL
CODE_REVIEWER_MODEL_PATH = "codeReviewer/models"
model, tokenizer, config = load_trained_model(CODE_REVIEWER_MODEL_PATH)



# Make a prediction
patch = [  
   """
@@ -5,2 +5,4 @@
+MAX_RETRIES = 3
+TIMEOUT = 30
class APIClient:
""",
   """
@@ -5,2 +5,2 @@
-def calculate_discount(price, discount):
+def calculate_discount(price: float, discount: float) -> float:
    return price * (1 - discount)
""",
'''@@ -10,7 +10,7 @@
 def hello():
    + import re
    print("Hello, world!")
''',
   """
@@ -22,6 +22,3 @@
def filter_numbers(numbers):
-    result = []
-    for num in numbers:
-        if num > 0:
-            result.append(num)
-    return result
+    return [num for num in numbers if num > 0]
""",
   """
@@ -30,5 +30,3 @@
def process_payment(amount):
-    print(f"Processing: {amount}")
    result = payment_gateway.charge(amount)
    return result
""",
   """
@@ -3,1 +3,2 @@
import json
+import logging
""",
   """
@@ -44,3 +44,3 @@
def divide(a, b):
-    return a / b
+    return a / b if b != 0 else None
"""
]



for i, diff in enumerate(patch):
    
    print("-" * 50)
    print(f"Prediction Result for {i+1}th patch:")
    print("-" * 50)
    print(f"Patch Text:\n{diff}\n")
    
    result = predict_code_review(model, tokenizer, diff)
    
    print(f"Prediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    
    if result['prediction_label'] == 'Review Needed':
        comment = generate_review_comment(diff)
        print(f"Generated Comment: {comment}")
        # print(f"[{i}] {comment}")
    
    print("-" * 50)
    print("\n\n")

    # print(f"Confidence: {result['confidence']:.2f}")