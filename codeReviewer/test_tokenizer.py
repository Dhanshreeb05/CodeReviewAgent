import os
from transformers import AutoTokenizer

def test_tokenizer():
    tokenizer_path = "models/tokenizer"
    
    print(f"Testing tokenizer loading from: {tokenizer_path}")
    print(f"Directory exists: {os.path.exists(tokenizer_path)}")
    
    if os.path.exists(tokenizer_path):
        print(f"Files in tokenizer directory:")
        for file in os.listdir(tokenizer_path):
            file_path = os.path.join(tokenizer_path, file)
            size = os.path.getsize(file_path)
            print(f"  {file}: {size} bytes")
    
    try:
        # Try loading the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✅ Tokenizer loaded successfully with {len(tokenizer)} tokens")
        return True
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return False

if __name__ == "__main__":
    test_tokenizer()