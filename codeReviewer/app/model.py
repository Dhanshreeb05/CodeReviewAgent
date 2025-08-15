import torch
import torch.nn as nn
import logging
import re
import json
import os
from typing import Tuple
from transformers import AutoTokenizer, AutoModel

class CodeReviewClassifier(nn.Module):
    """Your exact model architecture from the notebook"""
    
    def __init__(self, model_name='microsoft/codebert-base', num_numerical_features=8):
        super().__init__()
        
        # Load pre-trained CodeBERT
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

class QualityModelWrapper:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load your trained model exactly as you trained it"""
        try:
            self.logger.info(f"Loading model from {self.model_dir}")
            
            # Load model configuration
            with open(f"{self.model_dir}/model_config.json", 'r') as f:
                self.config = json.load(f)
            
            # rebuild tokenizer to avoid compatibility issues
            self.logger.info("Loading original CodeBERT tokenizer and adding special tokens")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            
            # Add special tokens from config
            special_tokens = self.config.get('special_tokens', ['[ADD]', '[DEL]', '[KEEP]', '[SEP]'])
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            
            self.logger.info(f"Tokenizer ready with {len(self.tokenizer)} tokens")
            
            # Create model with same architecture
            self.model = CodeReviewClassifier(
                model_name=self.config['model_name'],
                num_numerical_features=self.config['num_numerical_features']
            )
            
            # CRITICAL: Resize embeddings to match tokenizer
            self.model.codebert.resize_token_embeddings(len(self.tokenizer))
            self.logger.info(f"Resized embeddings to {len(self.tokenizer)} tokens")
            
            # Load trained weights
            self.model.load_state_dict(
                torch.load(f"{self.model_dir}/model_weights.pt", map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def preprocess_patch(self, patch_text: str) -> dict:
        """Preprocess exactly like in your training"""
        if not patch_text or patch_text.strip() == '':
            return {'processed_patch': '', 'features': [0.0] * 8}
        
        # Clean up whitespace
        patch = re.sub(r'\s+', ' ', patch_text)
        
        # Process each line
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
        
        processed_patch = ' '.join(processed_lines)
        
        # Extract numerical features
        patch_length = len(patch_text)
        num_additions = max(0, patch_text.count('+') - patch_text.count('@@'))
        num_deletions = max(0, patch_text.count('-') - patch_text.count('@@'))
        total_changes = num_additions + num_deletions
        
        # For API, we don't have msg, lang, proj info, so set defaults
        has_message = 0  # No message in API
        message_length = 0
        is_python = 0  # Unknown language
        is_undefined_lang = 1  # Treat as undefined
        
        numerical_features = [
            float(patch_length),
            float(num_additions),
            float(num_deletions),
            float(total_changes),
            float(has_message),
            float(message_length),
            float(is_python),
            float(is_undefined_lang)
        ]
        
        return {
            'processed_patch': processed_patch,
            'features': numerical_features
        }
    
    def predict(self, code_diff: str) -> Tuple[bool, float, str]:
        """Make prediction using your trained model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess input
            preprocessed = self.preprocess_patch(code_diff)
            
            # Tokenize
            encoding = self.tokenizer(
                preprocessed['processed_patch'],
                truncation=True,
                padding='max_length',
                max_length=256,  # Same as training
                return_tensors='pt'
            )
            
            # Prepare numerical features
            numerical_features = torch.tensor(
                preprocessed['features'], 
                dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            numerical_features = numerical_features.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, numerical_features)
                probabilities = torch.softmax(logits, dim=1)
                
                # Get probability of needing review (class 1)
                confidence = float(probabilities[0][1])
                needs_review = confidence > 0.5
            
            reasoning = self._generate_reasoning(needs_review, confidence)
            
            return needs_review, confidence, reasoning
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def _generate_reasoning(self, needs_review: bool, confidence: float) -> str:
        """Generate human-readable reasoning"""
        if needs_review:
            if confidence > 0.8:
                return f"Code likely needs review - high confidence ({confidence:.1%})"
            elif confidence > 0.6:
                return f"Code probably needs review - medium confidence ({confidence:.1%})"
            else:
                return f"Code may need review - low confidence ({confidence:.1%})"
        else:
            return f"Code appears clean and ready to merge ({confidence:.1%} confidence)"

# Global model instance
model_wrapper = QualityModelWrapper("models")