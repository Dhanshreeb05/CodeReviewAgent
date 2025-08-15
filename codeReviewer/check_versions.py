import transformers
import torch
import json

def check_versions():
    print("Current library versions:")
    print(f"  transformers: {transformers.__version__}")
    print(f"  torch: {torch.__version__}")
    
    # Check what version was used during training
    try:
        with open("models/training_results.json", 'r') as f:
            training_results = json.load(f)
            
        print("\nTraining environment (if recorded):")
        training_info = training_results.get('training_info', {})
        for key, value in training_info.items():
            if 'version' in key.lower():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"\nCouldn't read training results: {e}")
    
    # Check tokenizer config for version info
    try:
        with open("models/tokenizer/tokenizer_config.json", 'r') as f:
            tokenizer_config = json.load(f)
            
        print("\nTokenizer was saved with:")
        if 'transformers_version' in tokenizer_config:
            print(f"  transformers version: {tokenizer_config['transformers_version']}")
        else:
            print("  transformers version: Not recorded")
            
        print(f"  tokenizer_class: {tokenizer_config.get('tokenizer_class', 'Unknown')}")
        
    except Exception as e:
        print(f"\nCouldn't read tokenizer config: {e}")

if __name__ == "__main__":
    check_versions()