import torch
import logging
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class CommentModelWrapper:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load your trained CodeT5 model"""
        try:
            self.logger.info(f"Loading comment generation model from {self.model_dir}")
            
            # Load tokenizer - using Salesforce/codet5-large as base and fine-tuned model
            self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
            
            # Load fine-tuned model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Comment generation model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def format_diff_for_model(self, patch: str) -> str:
        """Format diff exactly like your training - same function from generateComment.py"""
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
    
    def generate_comment(self, code_diff: str) -> Tuple[str, float]:
        """Generate review comment exactly like your generateComment.py"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess the diff text
            formatted_diff = self.format_diff_for_model(code_diff)
            
            # Tokenize the input
            inputs = self.tokenizer(
                formatted_diff,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).input_ids
            
            # Move to device
            inputs = inputs.to(self.device)
            
            # Generate comment with parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=128,  
                    num_beams=5,     
                    early_stopping=True
                )
            
            # Decode the output
            comment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple confidence calculation
            confidence = self._calculate_confidence(comment, len(formatted_diff))
            
            return comment, confidence
            
        except Exception as e:
            self.logger.error(f"Comment generation failed: {e}")
            raise
    
    def _calculate_confidence(self, comment: str, input_length: int) -> float:
        """Simple confidence calculation - you can improve this"""
        if not comment or len(comment.strip()) < 5:
            return 0.3
        elif len(comment.split()) > 20:  # Very long comments might be verbose
            return 0.6
        elif len(comment.split()) < 3:   # Very short comments might be incomplete
            return 0.5
        else:
            return 0.8  # Good length comments

# Global model instance
model_wrapper = CommentModelWrapper("models")