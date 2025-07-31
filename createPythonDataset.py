import json
import pandas as pd
import numpy as np
from pathlib import Path

def read_jsonl_to_dataframe(file_path, max_rows=None):
    """Read JSONL file and convert to pandas DataFrame"""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
                
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {i+1}")
                    continue
    
    return pd.DataFrame(data)

def load_all_training_chunks(data_dir):
    """Load all training chunk files and combine into one DataFrame"""
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("cls-train-chunk-*.jsonl"))
    
    if not train_files:
        print("No training files found.")
        return pd.DataFrame()
    
    all_dfs = []
    for file_path in train_files:
        print(f"Loading {file_path.name}...")
        df = read_jsonl_to_dataframe(file_path)
        print(f"  Loaded {len(df)} rows")
        all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined training data: {len(combined_df):,} rows")
        return combined_df
    else:
        return pd.DataFrame()

def load_test_and_val_data(data_dir):
    """Load cls-test.jsonl and cls-valid.jsonl files"""
    data_dir = Path(data_dir)
    
    # Load test data
    test_path = data_dir / "cls-test.jsonl"
    if test_path.exists():
        print(f"Loading {test_path.name}...")
        test_df = read_jsonl_to_dataframe(test_path)
        print(f"  Loaded {len(test_df)} rows")
    else:
        print(f"Warning: {test_path} not found")
        test_df = pd.DataFrame()
    
    # Load validation data
    val_path = data_dir / "cls-valid.jsonl"
    if val_path.exists():
        print(f"Loading {val_path.name}...")
        val_df = read_jsonl_to_dataframe(val_path)
        print(f"  Loaded {len(val_df)} rows")
    else:
        print(f"Warning: {val_path} not found")
        val_df = pd.DataFrame()
    
    return test_df, val_df

def clean_and_filter_data(df, dataset_name):
    """Clean data and filter for Python and undefined languages"""
    print(f"\n=== PROCESSING {dataset_name.upper()} ===")
    print(f"Original data: {len(df):,} rows")
    
    if len(df) == 0:
        return df
    
    # Fill missing language data with 'undefined'
    if 'lang' in df.columns:
        df['lang'] = df['lang'].fillna('undefined')
    
    # Fill missing project data with 'unknown-project'
    if 'proj' in df.columns:
        df['proj'] = df['proj'].fillna('unknown-project')
    
    # Filter for Python and undefined only
    target_languages = ['py', 'undefined']
    filtered_df = df[df['lang'].isin(target_languages)].copy()
    
    print(f"Filtered data: {len(filtered_df):,} rows")
    
    # Show distribution
    if len(filtered_df) > 0:
        lang_counts = filtered_df['lang'].value_counts()
        print(f"Language distribution:")
        for lang, count in lang_counts.items():
            percentage = (count / len(filtered_df)) * 100
            readable_name = "Python" if lang == 'py' else "Undefined"
            print(f"  {lang} ({readable_name}): {count:,} ({percentage:.1f}%)")
        
        if 'y' in filtered_df.columns:
            label_counts = filtered_df['y'].value_counts()
            print(f"Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(filtered_df)) * 100
                meaning = "Review needed" if label == 1 else "No review needed"
                print(f"  {label} ({meaning}): {count:,} ({percentage:.1f}%)")
    
    return filtered_df

def save_dataset(df, filename, output_dir="./python_data"):
    """Save dataset in JSONL and pickle formats"""
    if len(df) == 0:
        print(f"Warning: {filename} is empty, skipping save")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as JSONL
    jsonl_path = output_path / f"{filename}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f)
            f.write('\n')
    
    # Save as pickle for fast loading
    pickle_path = output_path / f"{filename}.pkl"
    df.to_pickle(pickle_path)
    
    print(f"Saved {filename}: {len(df):,} rows")
    return jsonl_path, pickle_path

# Main execution
if __name__ == "__main__":
    print("=== CREATING PYTHON + UNDEFINED DATASET ===")
    print("Using existing test/val files, keeping all training data")
    
    # Load training data (all chunks combined)
    print("\n=== LOADING TRAINING DATA ===")
    train_df = load_all_training_chunks("./data")
    
    # Load test and validation data
    print("\n=== LOADING TEST AND VALIDATION DATA ===")
    test_df, val_df = load_test_and_val_data("./data")
    
    # Process each dataset
    train_filtered = clean_and_filter_data(train_df, "training")
    test_filtered = clean_and_filter_data(test_df, "test")
    val_filtered = clean_and_filter_data(val_df, "validation")
    
    # Save filtered datasets
    print(f"\n=== SAVING FILTERED DATASETS ===")
    
    if len(train_filtered) > 0:
        save_dataset(train_filtered, "train")
    
    if len(test_filtered) > 0:
        save_dataset(test_filtered, "test")
    
    if len(val_filtered) > 0:
        save_dataset(val_filtered, "val")
    
    # Save summary
    summary_path = Path("./python_data/dataset_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== PYTHON + UNDEFINED DATASET SUMMARY ===\n\n")
        
        f.write("Dataset sizes:\n")
        f.write(f"  train.jsonl: {len(train_filtered):,} rows\n")
        f.write(f"  test.jsonl:  {len(test_filtered):,} rows\n")
        f.write(f"  val.jsonl:   {len(val_filtered):,} rows\n")
        f.write(f"  Total:       {len(train_filtered) + len(test_filtered) + len(val_filtered):,} rows\n\n")
        
        f.write("Languages included: Python (py) and Undefined\n")
        f.write("Source files:\n")
        f.write("  - Training: cls-train-chunk-*.jsonl (all chunks combined)\n")
        f.write("  - Test: cls-test.jsonl\n")
        f.write("  - Validation: cls-valid.jsonl\n\n")
        
        if len(train_filtered) > 0:
            f.write("Available columns:\n")
            for col in train_filtered.columns:
                f.write(f"  {col}\n")
        
        f.write(f"\nLabel meanings:\n")
        f.write(f"  0 = No review needed\n")
        f.write(f"  1 = Review needed\n")
    
    print(f"Saved summary: {summary_path}")
    
    print(f"\n=== COMPLETE ===")
    print(f"Filtered datasets ready:")
    print(f"  train.jsonl: {len(train_filtered):,} rows")
    print(f"  test.jsonl:  {len(test_filtered):,} rows") 
    print(f"  val.jsonl:   {len(val_filtered):,} rows")
    print(f"\nAll files saved in ./python_data/ directory")