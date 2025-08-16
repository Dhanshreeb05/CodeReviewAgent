import json
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import difflib


class CodeReviewDatasetAnalyzer:
    def __init__(self, data_paths):
        """
        Initialize analyzer with paths to train, test, and validation files
        data_paths: dict with keys 'train', 'test', 'valid' and file paths as values
        """
        self.data_paths = data_paths
        self.datasets = {}
        self.combined_stats = {}

    def load_datasets(self):
        """Load all dataset files"""
        for split_name, file_path in self.data_paths.items():
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            self.datasets[split_name] = data
            print(f"Loaded {len(data)} samples from {split_name}")

    def extract_basic_statistics(self):
        """Extract basic dataset statistics"""
        stats = {
            'dataset_sizes': {},
            'total_samples': 0,
            'label_distribution': Counter(),
            'splits_info': {}
        }

        for split_name, data in self.datasets.items():
            stats['dataset_sizes'][split_name] = len(data)
            stats['total_samples'] += len(data)

            # Label distribution
            labels = [item.get('y', 'unknown') for item in data]
            stats['label_distribution'].update(labels)

            # Split-specific info
            stats['splits_info'][split_name] = {
                'size': len(data),
                'percentage': len(data) / sum(len(d) for d in self.datasets.values()) * 100
            }

        return stats

    def extract_text_statistics(self):
        """Extract statistics about text fields"""
        all_data = []
        for data in self.datasets.values():
            all_data.extend(data)

        # Message statistics
        messages = [item.get('msg', '') for item in all_data]
        message_lengths = [len(msg) for msg in messages]
        message_word_counts = [len(msg.split()) for msg in messages]

        # Original file statistics
        oldfs = [item.get('oldf', '') for item in all_data]
        oldf_lengths = [len(oldf) for oldf in oldfs]
        oldf_line_counts = [oldf.count('\n') for oldf in oldfs]

        # Patch statistics
        patches = [item.get('patch', '') for item in all_data]
        patch_lengths = [len(patch) for patch in patches]
        patch_line_counts = [patch.count('\n') for patch in patches]

        stats = {
            'message_stats': {
                'count': len(messages),
                'avg_length': np.mean(message_lengths),
                'median_length': np.median(message_lengths),
                'min_length': min(message_lengths),
                'max_length': max(message_lengths),
                'std_length': np.std(message_lengths),
                'avg_word_count': np.mean(message_word_counts),
                'empty_messages': sum(1 for msg in messages if not msg.strip())
            },
            'oldf_stats': {
                'avg_length': np.mean(oldf_lengths),
                'median_length': np.median(oldf_lengths),
                'avg_lines': np.mean(oldf_line_counts),
                'max_lines': max(oldf_line_counts),
                'empty_files': sum(1 for oldf in oldfs if not oldf.strip())
            },
            'patch_stats': {
                'avg_length': np.mean(patch_lengths),
                'median_length': np.median(patch_lengths),
                'avg_lines': np.mean(patch_line_counts),
                'empty_patches': sum(1 for patch in patches if not patch.strip())
            }
        }

        return stats

    def extract_code_statistics(self):
        """Extract code-specific statistics"""
        all_data = []
        for data in self.datasets.values():
            all_data.extend(data)

        # Programming language distribution
        languages = [item.get('lang', 'unknown') for item in all_data]
        lang_distribution = Counter(languages)

        # Project distribution
        projects = [item.get('proj', 'unknown') for item in all_data]
        project_distribution = Counter(projects)

        # Patch analysis
        patch_stats = self._analyze_patches(all_data)

        # File extension analysis (from oldf content)
        file_extensions = self._extract_file_extensions(all_data)

        stats = {
            'language_distribution': dict(lang_distribution),
            'project_distribution': dict(project_distribution),
            'patch_analysis': patch_stats,
            'file_extensions': dict(file_extensions),
            'unique_languages': len(lang_distribution),
            'unique_projects': len(project_distribution)
        }

        return stats

    def _analyze_patches(self, data):
        """Analyze patch content for code changes"""
        patch_stats = {
            'lines_added': [],
            'lines_removed': [],
            'files_changed': [],
            'hunks_per_patch': []
        }

        for item in data:
            patch = item.get('patch', '')
            if not patch:
                continue

            lines_added = patch.count('\n+') - patch.count('\n++')  # Exclude +++ lines
            lines_removed = patch.count('\n-') - patch.count('\n--')  # Exclude --- lines

            # Count hunks (starting with @@)
            hunks = patch.count('@@') // 2  # Each hunk has @@ at start and end

            # Estimate files changed (simple heuristic)
            files_changed = max(1, patch.count('+++'))

            patch_stats['lines_added'].append(lines_added)
            patch_stats['lines_removed'].append(lines_removed)
            patch_stats['files_changed'].append(files_changed)
            patch_stats['hunks_per_patch'].append(hunks)

        # Calculate summary statistics
        summary_stats = {}
        for key, values in patch_stats.items():
            if values:
                summary_stats[key] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'max': max(values),
                    'min': min(values),
                    'std': np.std(values)
                }

        return summary_stats

    def _extract_file_extensions(self, data):
        """Extract file extensions from patches or infer from language"""
        extensions = []

        for item in data:
            lang = item.get('lang', '')
            patch = item.get('patch', '')

            # Try to extract from patch headers
            if '---' in patch and '+++' in patch:
                lines = patch.split('\n')
                for line in lines:
                    if line.startswith('+++') or line.startswith('---'):
                        # Extract filename and extension
                        filename = line.split('\t')[0][4:] if '\t' in line else line[4:]
                        if '.' in filename:
                            ext = filename.split('.')[-1]
                            if len(ext) <= 5:  # Reasonable extension length
                                extensions.append(ext)
                                break

            # Fallback to language mapping
            elif lang:
                lang_to_ext = {
                    'py': 'py', 'python': 'py',
                    'js': 'js', 'javascript': 'js',
                    'java': 'java',
                    'cpp': 'cpp', 'c++': 'cpp',
                    'c': 'c',
                    'rb': 'rb', 'ruby': 'rb',
                    'go': 'go',
                    'rs': 'rs', 'rust': 'rs'
                }
                extensions.append(lang_to_ext.get(lang.lower(), lang))

        return Counter(extensions)

    def extract_content_analysis(self):
        """Analyze content patterns in review messages"""
        all_data = []
        for data in self.datasets.values():
            all_data.extend(data)

        messages = [item.get('msg', '') for item in all_data if item.get('msg')]

        # Common words analysis
        all_words = []
        for msg in messages:
            words = re.findall(r'\b\w+\b', msg.lower())
            all_words.extend(words)

        word_freq = Counter(all_words)

        # Message categories (simple keyword-based classification)
        categories = self._categorize_messages(messages)

        # Sentiment analysis (simple approach)
        sentiment_stats = self._analyze_sentiment(messages)

        stats = {
            'vocabulary_size': len(word_freq),
            'most_common_words': dict(word_freq.most_common(20)),
            'message_categories': dict(categories),
            'sentiment_analysis': sentiment_stats,
            'avg_unique_words_per_message': np.mean([len(set(msg.lower().split())) for msg in messages])
        }

        return stats

    def _categorize_messages(self, messages):
        """Categorize review messages based on keywords"""
        categories = defaultdict(int)

        category_keywords = {
            'style': ['style', 'format', 'formatting', 'indent', 'spacing', 'lint'],
            'bug': ['bug', 'error', 'fix', 'issue', 'problem', 'wrong'],
            'performance': ['performance', 'optimize', 'speed', 'memory', 'efficient'],
            'refactor': ['refactor', 'clean', 'simplify', 'restructure'],
            'documentation': ['comment', 'document', 'doc', 'explain', 'clarify'],
            'security': ['security', 'vulnerable', 'safe', 'validate', 'sanitize'],
            'test': ['test', 'testing', 'coverage', 'unit test'],
            'import': ['import', 'dependency', 'module', 'package']
        }

        for msg in messages:
            msg_lower = msg.lower()
            for category, keywords in category_keywords.items():
                if any(keyword in msg_lower for keyword in keywords):
                    categories[category] += 1
                    break
            else:
                categories['other'] += 1

        return categories

    def _analyze_sentiment(self, messages):
        """Simple sentiment analysis based on keywords"""
        positive_words = ['good', 'great', 'nice', 'perfect', 'excellent', 'correct', 'right']
        negative_words = ['bad', 'wrong', 'error', 'issue', 'problem', 'incorrect', 'fail']

        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

        for msg in messages:
            msg_lower = msg.lower()
            pos_count = sum(1 for word in positive_words if word in msg_lower)
            neg_count = sum(1 for word in negative_words if word in msg_lower)

            if pos_count > neg_count:
                sentiment_counts['positive'] += 1
            elif neg_count > pos_count:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1

        return sentiment_counts

    def extract_quality_metrics(self):
        """Extract data quality metrics"""
        all_data = []
        for data in self.datasets.values():
            all_data.extend(data)

        required_fields = ['oldf', 'patch', 'msg', 'id', 'y']
        quality_stats = {
            'total_samples': len(all_data),
            'complete_samples': 0,
            'missing_fields': defaultdict(int),
            'empty_fields': defaultdict(int),
            'duplicate_ids': 0,
            'duplicate_messages': 0
        }

        ids_seen = set()
        messages_seen = Counter()

        for item in all_data:
            # Check completeness
            complete = True
            for field in required_fields:
                if field not in item:
                    quality_stats['missing_fields'][field] += 1
                    complete = False
                elif not str(item[field]).strip():
                    quality_stats['empty_fields'][field] += 1
                    complete = False

            if complete:
                quality_stats['complete_samples'] += 1

            # Check for duplicate IDs
            item_id = item.get('id')
            if item_id in ids_seen:
                quality_stats['duplicate_ids'] += 1
            ids_seen.add(item_id)

            # Count message duplicates
            msg = item.get('msg', '').strip()
            if msg:
                messages_seen[msg] += 1

        quality_stats['duplicate_messages'] = sum(1 for count in messages_seen.values() if count > 1)
        quality_stats['completeness_rate'] = quality_stats['complete_samples'] / len(all_data) * 100

        return quality_stats

    def generate_full_report(self):
        """Generate comprehensive dataset analysis report"""
        print("Loading datasets...")
        self.load_datasets()

        print("Extracting statistics...")
        basic_stats = self.extract_basic_statistics()
        text_stats = self.extract_text_statistics()
        code_stats = self.extract_code_statistics()
        content_stats = self.extract_content_analysis()
        quality_stats = self.extract_quality_metrics()

        # Combine all statistics
        full_stats = {
            'basic_statistics': basic_stats,
            'text_statistics': text_stats,
            'code_statistics': code_stats,
            'content_analysis': content_stats,
            'quality_metrics': quality_stats
        }

        return full_stats

    def print_summary_report(self, stats):
        """Print a formatted summary of the statistics"""
        print("\n" + "=" * 60)
        print("CODE REVIEW DATASET ANALYSIS REPORT")
        print("=" * 60)

        # Basic Statistics
        print("\n📊 BASIC STATISTICS")
        print("-" * 30)
        basic = stats['basic_statistics']
        print(f"Total samples: {basic['total_samples']:,}")
        for split, size in basic['dataset_sizes'].items():
            percentage = basic['splits_info'][split]['percentage']
            print(f"  {split.capitalize()}: {size:,} ({percentage:.1f}%)")

        print(f"\nLabel distribution:")
        for label, count in basic['label_distribution'].items():
            print(f"  Label {label}: {count:,}")

        # Text Statistics
        print("\n📝 TEXT STATISTICS")
        print("-" * 30)
        text = stats['text_statistics']
        msg_stats = text['message_stats']
        print(f"Average message length: {msg_stats['avg_length']:.1f} chars")
        print(f"Average words per message: {msg_stats['avg_word_count']:.1f}")
        print(f"Empty messages: {msg_stats['empty_messages']}")

        oldf_stats = text['oldf_stats']
        print(f"Average file length: {oldf_stats['avg_length']:,.0f} chars")
        print(f"Average lines per file: {oldf_stats['avg_lines']:.1f}")

        # Code Statistics
        print("\n💻 CODE STATISTICS")
        print("-" * 30)
        code = stats['code_statistics']
        print(f"Unique languages: {code['unique_languages']}")
        print(f"Unique projects: {code['unique_projects']}")

        print("\nTop 5 languages:")
        lang_items = sorted(code['language_distribution'].items(), key=lambda x: x[1], reverse=True)
        for lang, count in lang_items[:5]:
            print(f"  {lang}: {count}")

        # Content Analysis
        print("\n🔍 CONTENT ANALYSIS")
        print("-" * 30)
        content = stats['content_analysis']
        print(f"Vocabulary size: {content['vocabulary_size']:,}")

        print("\nMessage categories:")
        for category, count in sorted(content['message_categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category.capitalize()}: {count}")

        # Quality Metrics
        print("\n✅ QUALITY METRICS")
        print("-" * 30)
        quality = stats['quality_metrics']
        print(f"Completeness rate: {quality['completeness_rate']:.1f}%")
        print(f"Duplicate IDs: {quality['duplicate_ids']}")
        print(f"Duplicate messages: {quality['duplicate_messages']}")

        if quality['missing_fields']:
            print("\nMissing fields:")
            for field, count in quality['missing_fields'].items():
                print(f"  {field}: {count}")

    def save_stats_to_file(self, stats, filename='dataset_stats.json'):
        """Save statistics to JSON file"""

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        clean_stats = convert_numpy_types(stats)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clean_stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Define paths to your dataset files
    data_paths = {
        'train': 'msg-train.jsonl',
        'test': 'msg-test.jsonl',
        'valid': 'msg-valid.jsonl'
    }

    # Initialize analyzer
    analyzer = CodeReviewDatasetAnalyzer(data_paths)

    # Generate full analysis
    stats = analyzer.generate_full_report()

    # Print summary
    analyzer.print_summary_report(stats)

    # Save detailed stats to file
    analyzer.save_stats_to_file(stats, 'detailed_dataset_stats.json')

    # You can also access individual statistics:
    # print("\nDetailed message statistics:")
    # print(stats['text_statistics']['message_stats'])