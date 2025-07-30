import json
import pandas as pd
import numpy as np
from pathlib import Path

def read_jsonl_to_dataframe(file_path, max_rows=None):
    """
    Read JSONL file and convert to pandas DataFrame
    
    Args:
        file_path: Path to JSONL file
        max_rows: Maximum number of rows to read (None for all)
    
    Returns:
        pandas DataFrame
    """
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

def load_all_training_chunks(data_dir, max_rows_per_file=None):
    """
    Load all training chunk files and combine into one DataFrame
    """
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("cls-train-chunk-*.jsonl"))
    
    if not train_files:
        print("No training files found.")
        return pd.DataFrame()
    
    all_dfs = []
    for file_path in train_files:
        print(f"Loading {file_path.name}...")
        df = read_jsonl_to_dataframe(file_path, max_rows_per_file)
        print(f"  Loaded {len(df)} rows")
        all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined training data: {len(combined_df):,} rows")
        return combined_df
    else:
        return pd.DataFrame()

# Base language mapping - we'll expand this automatically
BASE_LANGUAGE_MAP = {
    'py': 'python',
    'java': 'java', 
    'js': 'javascript',
    'cpp': 'cpp',
    'go': 'go',
    '.cs': 'csharp',
    'rb': 'ruby',
    'c': 'c',
    'php': 'php'
}

def discover_all_languages(df):
    """Discover all unique language values in the dataset and create complete mapping"""
    print(f"\n=== DISCOVERING ALL LANGUAGES ===")
    
    if 'lang' not in df.columns:
        print("No 'lang' column found")
        return BASE_LANGUAGE_MAP
    
    # Get all unique language values (including NaN)
    all_langs = df['lang'].value_counts(dropna=False)
    
    print(f"Found {len(all_langs)} unique language values:")
    for lang, count in all_langs.items():
        percentage = (count / len(df)) * 100
        if pd.isna(lang):
            print(f"  NaN/Missing: {count:,} ({percentage:.1f}%)")
        else:
            print(f"  '{lang}': {count:,} ({percentage:.1f}%)")
    
    # Create complete language mapping
    complete_map = BASE_LANGUAGE_MAP.copy()
    
    # Add any missing languages with automatic mapping
    for lang in all_langs.index:
        if pd.isna(lang):
            complete_map['undefined'] = 'undefined'  # For NaN values
        elif lang not in complete_map:
            # Auto-generate readable name for unknown languages
            if lang.startswith('.'):
                # File extension like .cs, .py, etc.
                readable_name = lang[1:] if len(lang) > 1 else 'unknown'
            else:
                readable_name = lang
            complete_map[lang] = readable_name
            print(f"  Added new language mapping: '{lang}' -> '{readable_name}'")
    
    print(f"\nComplete language mapping:")
    for code, name in sorted(complete_map.items()):
        print(f"  {code} -> {name}")
    
    return complete_map

def clean_and_fill_missing_data(df, language_map):
    """Clean data and fill missing language values with 'undefined'"""
    print(f"\n=== CLEANING DATA ===")
    print(f"Original data: {len(df):,} rows")
    
    # Fill missing language data with 'undefined'
    if 'lang' in df.columns:
        missing_lang_count = df['lang'].isna().sum()
        if missing_lang_count > 0:
            print(f"Missing language data: {missing_lang_count:,} rows")
            df['lang'] = df['lang'].fillna('undefined')
            print(f"Filled missing language data with 'undefined'")
    
    # Fill missing project data with 'unknown-project'
    if 'proj' in df.columns:
        missing_proj_count = df['proj'].isna().sum()
        if missing_proj_count > 0:
            print(f"Missing project data: {missing_proj_count:,} rows")
            df['proj'] = df['proj'].fillna('unknown-project')
            print(f"Filled missing project data with 'unknown-project'")
    
    return df

def analyze_all_data_including_undefined(df, language_map):
    """Analyze all data using the complete language mapping"""
    print(f"\n=== ANALYSIS OF ALL DATA ===")
    print(f"Total samples: {len(df):,}")
    
    # Language distribution
    if 'lang' in df.columns:
        print(f"\nLanguage distribution:")
        lang_counts = df['lang'].value_counts()
        total = len(df)
        
        for lang, count in lang_counts.items():
            readable_name = language_map.get(lang, f"UNKNOWN({lang})")
            percentage = (count / total) * 100
            print(f"  {lang} ({readable_name}): {count:,} ({percentage:.1f}%)")
    
    # Label distribution
    if 'y' in df.columns:
        print(f"\nLabel distribution:")
        label_counts = df['y'].value_counts()
        total = len(df)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            meaning = "Review needed" if label == 1 else "No review needed"
            print(f"  {label} ({meaning}): {count:,} ({percentage:.1f}%)")
    
    # Language vs Label breakdown
    if 'lang' in df.columns and 'y' in df.columns:
        print(f"\nLanguage vs Label breakdown:")
        lang_label_cross = pd.crosstab(df['lang'], df['y'], margins=True)
        print(lang_label_cross)
        
        # Percentage breakdown by language
        print(f"\nPercentage needing review by language:")
        for lang in sorted(df['lang'].unique()):
            lang_data = df[df['lang'] == lang]
            if len(lang_data) > 0:
                review_needed = len(lang_data[lang_data['y'] == 1])
                no_review = len(lang_data[lang_data['y'] == 0])
                total_lang = len(lang_data)
                percentage = (review_needed / total_lang) * 100
                readable_name = language_map.get(lang, f"UNKNOWN({lang})")
                print(f"  {lang} ({readable_name}): {review_needed:,}/{total_lang:,} ({percentage:.1f}%) need review")

def filter_by_language_code(df, lang_codes, include_undefined=False):
    """
    Filter DataFrame by language codes
    
    Args:
        df: pandas DataFrame
        lang_codes: String or list of language codes to filter by
        include_undefined: Whether to include undefined language samples
    
    Returns:
        Filtered DataFrame
    """
    if isinstance(lang_codes, str):
        lang_codes = [lang_codes]
    
    if include_undefined and 'undefined' not in lang_codes:
        lang_codes = lang_codes + ['undefined']
    
    if 'lang' not in df.columns:
        print("No 'lang' column found")
        return df
    
    filtered_df = df[df['lang'].isin(lang_codes)]
    print(f"Filtered from {len(df):,} to {len(filtered_df):,} rows for languages: {lang_codes}")
    return filtered_df

# Example usage:
if __name__ == "__main__":
    # Load FULL training data 
    print("=== Loading FULL training data ===")
    print("Warning: This will load ~266k samples (~11GB). This may take a few minutes...")
    
    full_train_df = load_all_training_chunks("./data")
    
    if len(full_train_df) > 0:
        # STEP 1: Discover all languages in the dataset
        complete_language_map = discover_all_languages(full_train_df)
        
        # STEP 2: Clean and fill missing data
        full_train_df = clean_and_fill_missing_data(full_train_df, complete_language_map)
        
        # STEP 3: Analyze all data using complete mapping
        analyze_all_data_including_undefined(full_train_df, complete_language_map)
        
        # Project distribution (top 20)
        print(f"\nTop 20 projects by sample count:")
        proj_counts = full_train_df['proj'].value_counts().head(20)
        total = len(full_train_df)
        for proj, count in proj_counts.items():
            percentage = (count / total) * 100
            print(f"  {proj}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nDataset loaded and cleaned successfully!")
        
        # Show final language mapping used
        print(f"\n=== FINAL LANGUAGE MAPPING USED ===")
        for code, name in sorted(complete_language_map.items()):
            print(f"  {code} -> {name}")
        
        # Example filtering with complete language list
        print(f"\n=== Example Filtering ===")
        
        # Get all defined language codes (excluding undefined)
        defined_langs = [lang for lang in complete_language_map.keys() if lang != 'undefined']
        print(f"All defined languages: {defined_langs}")
        
        # Python only
        if 'py' in complete_language_map:
            python_data = filter_by_language_code(full_train_df, "py")
        
        # Undefined language only
        if 'undefined' in complete_language_map:
            undefined_data = filter_by_language_code(full_train_df, "undefined")
        
        # Save cleaned dataset with language mapping
        # full_train_df.to_pickle("cleaned_training_data.pkl")
        # with open("language_mapping.json", "w") as f:
        #     json.dump(complete_language_map, f, indent=2)
        # print("Saved cleaned dataset and language mapping!")
        
    else:
        print("No training data was loaded. Check your file paths and try again.")