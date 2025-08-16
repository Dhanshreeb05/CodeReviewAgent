import json
import re
from collections import defaultdict, Counter
import pandas as pd


def analyze_patch_patterns(data_paths):
    """Analyze common patterns in code patches"""

    all_patches = []
    for file_path in data_paths.values():
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('patch'):
                    all_patches.append(item['patch'])

    patterns = {
        'import_changes': 0,
        'function_definitions': 0,
        'variable_assignments': 0,
        'comment_changes': 0,
        'whitespace_changes': 0,
        'string_literals': 0,
        'method_calls': 0
    }

    for patch in all_patches:
        # Import statements
        if re.search(r'[\+\-]\s*(import|from .* import)', patch):
            patterns['import_changes'] += 1

        # Function definitions
        if re.search(r'[\+\-]\s*def\s+\w+', patch):
            patterns['function_definitions'] += 1

        # Variable assignments
        if re.search(r'[\+\-]\s*\w+\s*=', patch):
            patterns['variable_assignments'] += 1

        # Comments
        if re.search(r'[\+\-]\s*#|[\+\-]\s*/\*|[\+\-]\s*\*|[\+\-]\s*//', patch):
            patterns['comment_changes'] += 1

        # Whitespace only changes
        if re.search(r'^[\+\-]\s*$', patch, re.MULTILINE):
            patterns['whitespace_changes'] += 1

        # String literals
        if re.search(r'[\+\-].*["\'].*["\']', patch):
            patterns['string_literals'] += 1

        # Method calls
        if re.search(r'[\+\-].*\w+\(.*\)', patch):
            patterns['method_calls'] += 1

    return patterns


def analyze_message_complexity(data_paths):
    """Analyze complexity and readability of review messages"""

    all_messages = []
    for file_path in data_paths.values():
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('msg'):
                    all_messages.append(item['msg'])

    complexity_metrics = {
        'avg_sentence_length': 0,
        'avg_syllables_per_word': 0,
        'readability_score': 0,
        'technical_terms': 0,
        'question_messages': 0,
        'imperative_messages': 0,
        'code_snippets': 0
    }

    sentence_lengths = []
    technical_terms = set(['refactor', 'optimize', 'deprecated', 'async', 'sync',
                           'algorithm', 'performance', 'memory', 'thread', 'exception',
                           'inheritance', 'polymorphism', 'encapsulation', 'abstraction'])

    for msg in all_messages:
        # Sentence length
        sentences = re.split(r'[.!?]+', msg)
        for sentence in sentences:
            words = len(sentence.split())
            if words > 0:
                sentence_lengths.append(words)

        # Technical terms
        msg_lower = msg.lower()
        if any(term in msg_lower for term in technical_terms):
            complexity_metrics['technical_terms'] += 1

        # Question messages
        if '?' in msg:
            complexity_metrics['question_messages'] += 1

        # Imperative messages (starting with action words)
        imperative_words = ['use', 'add', 'remove', 'change', 'fix', 'update', 'check']
        first_word = msg.strip().split()[0].lower() if msg.strip() else ''
        if first_word in imperative_words:
            complexity_metrics['imperative_messages'] += 1

        # Code snippets (text in backticks or containing code-like patterns)
        if '`' in msg or re.search(r'\b\w+\(\)|->|=>', msg):
            complexity_metrics['code_snippets'] += 1

    complexity_metrics['avg_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    return complexity_metrics


def analyze_review_effectiveness(data_paths):
    """Analyze patterns that might indicate review effectiveness"""

    all_data = []
    for file_path in data_paths.values():
        with open(file_path, 'r') as f:
            for line in f:
                all_data.append(json.loads(line.strip()))

    effectiveness_metrics = {
        'specific_suggestions': 0,
        'general_feedback': 0,
        'positive_feedback': 0,
        'constructive_criticism': 0,
        'code_examples': 0,
        'links_to_docs': 0,
        'security_concerns': 0,
        'performance_notes': 0
    }

    for item in all_data:
        msg = item.get('msg', '').lower()

        # Specific suggestions (contains "should", "could", "try")
        if any(word in msg for word in ['should', 'could', 'try', 'consider', 'suggest']):
            effectiveness_metrics['specific_suggestions'] += 1
        else:
            effectiveness_metrics['general_feedback'] += 1

        # Positive feedback
        if any(word in msg for word in ['good', 'great', 'nice', 'well', 'correct']):
            effectiveness_metrics['positive_feedback'] += 1

        # Constructive criticism
        if any(word in msg for word in ['however', 'but', 'although', 'instead']):
            effectiveness_metrics['constructive_criticism'] += 1

        # Code examples (backticks or code patterns)
        if '`' in msg or '->' in msg or '=>' in msg:
            effectiveness_metrics['code_examples'] += 1

        # Links to documentation
        if 'http' in msg or 'docs' in msg or 'documentation' in msg:
            effectiveness_metrics['links_to_docs'] += 1

        # Security concerns
        if any(word in msg for word in ['security', 'vulnerable', 'safe', 'validate', 'sanitize']):
            effectiveness_metrics['security_concerns'] += 1

        # Performance notes
        if any(word in msg for word in ['performance', 'speed', 'memory', 'optimize', 'efficient']):
            effectiveness_metrics['performance_notes'] += 1

    return effectiveness_metrics


def generate_dataset_summary_table(stats):
    """Generate a summary table of key metrics"""

    summary_data = []

    # Basic metrics
    basic = stats['basic_statistics']
    summary_data.append(['Total Samples', f"{basic['total_samples']:,}"])
    summary_data.append(['Training Samples', f"{basic['dataset_sizes'].get('train', 0):,}"])
    summary_data.append(['Test Samples', f"{basic['dataset_sizes'].get('test', 0):,}"])
    summary_data.append(['Validation Samples', f"{basic['dataset_sizes'].get('valid', 0):,}"])

    # Text metrics
    text = stats['text_statistics']
    summary_data.append(['Avg Message Length', f"{text['message_stats']['avg_length']:.1f} chars"])
    summary_data.append(['Avg Words per Message', f"{text['message_stats']['avg_word_count']:.1f}"])
    summary_data.append(['Vocabulary Size', f"{stats['content_analysis']['vocabulary_size']:,}"])

    # Code metrics
    code = stats['code_statistics']
    summary_data.append(['Unique Languages', f"{code['unique_languages']}"])
    summary_data.append(['Unique Projects', f"{code['unique_projects']}"])

    # Quality metrics
    quality = stats['quality_metrics']
    summary_data.append(['Completeness Rate', f"{quality['completeness_rate']:.1f}%"])
    summary_data.append(['Duplicate Messages', f"{quality['duplicate_messages']}"])

    # Create DataFrame for nice formatting
    df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    return df


def export_statistics_report(stats, output_file='dataset_report.txt'):
    """Export a comprehensive text report"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CODE REVIEW DATASET ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Dataset Overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        basic = stats['basic_statistics']
        f.write(f"Total Samples: {basic['total_samples']:,}\n")
        for split, size in basic['dataset_sizes'].items():
            percentage = basic['splits_info'][split]['percentage']
            f.write(f"  {split.capitalize()}: {size:,} ({percentage:.1f}%)\n")
        f.write("\n")

        # Text Statistics
        f.write("TEXT STATISTICS\n")
        f.write("-" * 20 + "\n")
        text = stats['text_statistics']
        msg_stats = text['message_stats']
        f.write(f"Message Statistics:\n")
        f.write(f"  Average length: {msg_stats['avg_length']:.1f} characters\n")
        f.write(f"  Average words: {msg_stats['avg_word_count']:.1f}\n")
        f.write(f"  Min length: {msg_stats['min_length']} characters\n")
        f.write(f"  Max length: {msg_stats['max_length']} characters\n")
        f.write(f"  Empty messages: {msg_stats['empty_messages']}\n\n")

        # Language Distribution
        f.write("PROGRAMMING LANGUAGES\n")
        f.write("-" * 20 + "\n")
        code = stats['code_statistics']
        lang_dist = code['language_distribution']
        sorted_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs[:10]:  # Top 10
            f.write(f"  {lang}: {count}\n")
        f.write("\n")

        # Message Categories
        f.write("MESSAGE CATEGORIES\n")
        f.write("-" * 20 + "\n")
        content = stats['content_analysis']
        categories = content['message_categories']
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {category.capitalize()}: {count}\n")
        f.write("\n")

        # Quality Assessment
        f.write("DATA QUALITY\n")
        f.write("-" * 20 + "\n")
        quality = stats['quality_metrics']
        f.write(f"Completeness rate: {quality['completeness_rate']:.1f}%\n")
        f.write(f"Duplicate IDs: {quality['duplicate_ids']}\n")
        f.write(f"Duplicate messages: {quality['duplicate_messages']}\n")

        if quality['missing_fields']:
            f.write("Missing fields:\n")
            for field, count in quality['missing_fields'].items():
                f.write(f"  {field}: {count}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated successfully!\n")

    print(f"Detailed report saved to {output_file}")


# Main execution function for additional analysis
def run_additional_analysis(data_paths):
    """Run additional analysis beyond basic statistics"""

    print("Running additional analysis...")

    # Patch patterns
    print("Analyzing patch patterns...")
    patch_patterns = analyze_patch_patterns(data_paths)

    # Message complexity
    print("Analyzing message complexity...")
    message_complexity = analyze_message_complexity(data_paths)

    # Review effectiveness
    print("Analyzing review effectiveness...")
    review_effectiveness = analyze_review_effectiveness(data_paths)

    # Combine results
    additional_stats = {
        'patch_patterns': patch_patterns,
        'message_complexity': message_complexity,
        'review_effectiveness': review_effectiveness
    }

    # Print results
    print("\n" + "=" * 50)
    print("ADDITIONAL ANALYSIS RESULTS")
    print("=" * 50)

    print("\n📝 PATCH PATTERNS")
    print("-" * 20)
    for pattern, count in patch_patterns.items():
        print(f"{pattern.replace('_', ' ').title()}: {count}")

    print("\n🧠 MESSAGE COMPLEXITY")
    print("-" * 20)
    for metric, value in message_complexity.items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")

    print("\n✅ REVIEW EFFECTIVENESS")
    print("-" * 20)
    for metric, count in review_effectiveness.items():
        print(f"{metric.replace('_', ' ').title()}: {count}")

    return additional_stats


# Example usage
if __name__ == "__main__":
    data_paths = {
        'train': 'msg-train.jsonl',
        'test': 'msg-test.jsonl',
        'valid': 'msg-valid.jsonl'
    }

    # Run additional analysis
    additional_stats = run_additional_analysis(data_paths)

    # Save additional stats
    with open('additional_analysis.json', 'w') as f:
        json.dump(additional_stats, f, indent=2)

    print("\nAdditional analysis complete!")