import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

class CodeReviewClassifier(nn.Module):
    def __init__(self, model_name='microsoft/codebert-base', num_numerical_features=8):
        super().__init__()
        
        # Load pre-trained CodeBERT
        self.codebert = AutoModel.from_pretrained(model_name)
        self.codebert_dim = self.codebert.config.hidden_size
        
        # Numerical features processor
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Final classifier
        combined_dim = self.codebert_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, input_ids, attention_mask, numerical_features):
        # Process with CodeBERT
        codebert_output = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        code_features = codebert_output.last_hidden_state[:, 0, :]
        
        # Process numerical features
        numerical_processed = self.numerical_processor(numerical_features)
        
        # Combine and classify
        combined_features = torch.cat([code_features, numerical_processed], dim=1)
        logits = self.classifier(combined_features)
        
        return logits

def test_model_loading():
    print("Testing model loading...")
    
    model_dir = "models"
    
    try:
        # Load config
        with open(f"{model_dir}/model_config.json", 'r') as f:
            config = json.load(f)
        print("✅ Config loaded")
        print(f"   Config: {config}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
        print(f"✅ Tokenizer loaded with {len(tokenizer)} tokens")
        
        # Create model
        model = CodeReviewClassifier(
            model_name=config['model_name'],
            num_numerical_features=config['num_numerical_features']
        )
        print("✅ Model architecture created")
        
        # 🔧 FIX: Resize embeddings to match tokenizer size
        model.codebert.resize_token_embeddings(len(tokenizer))
        print(f"✅ Resized embeddings to {len(tokenizer)} tokens")
        
        # Load weights
        device = torch.device("cpu")
        model.load_state_dict(torch.load(f"{model_dir}/model_weights.pt", map_location=device))
        model.eval()
        print("✅ Model weights loaded")
        
        # Test prediction with dummy data
        sample_text = "[ADD] print('hello world')"
        encoding = tokenizer(
            sample_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # Dummy numerical features
        numerical_features = torch.tensor([[100.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'], numerical_features)
            probabilities = torch.softmax(logits, dim=1)
            
        print("✅ Test prediction successful!")
        print(f"   Logits: {logits}")
        print(f"   Probabilities: {probabilities}")
        print(f"   Prediction: {'Review needed' if probabilities[0][1] > 0.5 else 'No review needed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model is ready for the FastAPI service!")
    else:
        print("\n❌ Fix the issues above before proceeding")