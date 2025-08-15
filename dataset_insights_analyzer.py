import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DatasetInsightsAnalyzer:
    def __init__(self, data_dir="./data", output_dir="./dataset_insights"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_jsonl_file(self, file_path, max_rows=None):
        """Load JSONL file into DataFrame"""
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
                        continue
        return pd.DataFrame(data)
    
    def load_all_data(self, sample_size=None):
        """Load all training, test, and validation data"""
        print("Loading dataset files...")
        
        # Load training chunks
        train_files = sorted(self.data_dir.glob("cls-train-chunk-*.jsonl"))
        train_dfs = []
        
        for file_path in train_files:
            print(f"  Loading {file_path.name}...")
            df = self.load_jsonl_file(file_path, sample_size)
            train_dfs.append(df)
        
        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        
        # Load test and validation
        test_df = self.load_jsonl_file(self.data_dir / "cls-test.jsonl")
        val_df = self.load_jsonl_file(self.data_dir / "cls-valid.jsonl")
        
        print(f"Loaded: Train={len(train_df):,}, Test={len(test_df):,}, Val={len(val_df):,}")
        return train_df, test_df, val_df
    
    def clean_data(self, df):
        """Basic data cleaning"""
        if len(df) == 0:
            return df
        
        # Fill missing values
        df['lang'] = df['lang'].fillna('undefined')
        df['proj'] = df['proj'].fillna('unknown-project')
        df['msg'] = df['msg'].fillna('')
        df['patch'] = df['patch'].fillna('')
        
        return df
    
    def filter_python_undefined(self, df):
        """Filter for Python and undefined languages only"""
        if len(df) == 0:
            return df
        return df[df['lang'].isin(['py', 'undefined'])].copy()
    
    def extract_features(self, df):
        """Extract numerical features from the dataset"""
        if len(df) == 0:
            return df
        
        df = df.copy()
        
        # Patch statistics
        df['patch_length'] = df['patch'].str.len()
        df['num_additions'] = df['patch'].str.count(r'\+[^@]')
        df['num_deletions'] = df['patch'].str.count(r'\-[^@]')
        df['total_changes'] = df['num_additions'] + df['num_deletions']
        
        # Message statistics
        df['has_message'] = (df['msg'].str.len() > 0).astype(int)
        df['message_length'] = df['msg'].str.len()
        
        # Language indicators
        df['is_python'] = (df['lang'] == 'py').astype(int)
        df['is_undefined_lang'] = (df['lang'] == 'undefined').astype(int)
        
        return df
    
    def analyze_original_data(self, train_df, test_df, val_df):
        """Analyze the original unfiltered data"""
        print("\n" + "="*60)
        print("ORIGINAL DATASET ANALYSIS")
        print("="*60)
        
        # Combine all data for overall analysis
        all_data = pd.concat([train_df, test_df, val_df], ignore_index=True)
        all_data = self.clean_data(all_data)
        
        print(f"Total samples: {len(all_data):,}")
        
        # Language distribution
        if 'lang' in all_data.columns:
            lang_dist = all_data['lang'].value_counts()
            print(f"\nLanguage Distribution (Top 10):")
            for i, (lang, count) in enumerate(lang_dist.head(10).items()):
                print(f"  {lang}: {count:,} ({count/len(all_data)*100:.1f}%)")
        
        # Label distribution
        if 'y' in all_data.columns:
            label_dist = all_data['y'].value_counts()
            print(f"\nLabel Distribution:")
            for label, count in label_dist.items():
                meaning = "Review needed" if label == 1 else "No review needed"
                print(f"  {label} ({meaning}): {count:,} ({count/len(all_data)*100:.1f}%)")
        
        return all_data
    
    def analyze_filtered_data(self, train_df, test_df, val_df):
        """Analyze the filtered Python + undefined data"""
        print("\n" + "="*60)
        print("FILTERED DATASET ANALYSIS (Python + Undefined)")
        print("="*60)
        
        # Filter each split
        train_filtered = self.filter_python_undefined(self.clean_data(train_df))
        test_filtered = self.filter_python_undefined(self.clean_data(test_df))
        val_filtered = self.filter_python_undefined(self.clean_data(val_df))
        
        # Extract features
        train_filtered = self.extract_features(train_filtered)
        test_filtered = self.extract_features(test_filtered)
        val_filtered = self.extract_features(val_filtered)
        
        print(f"Filtered samples:")
        print(f"  Train: {len(train_filtered):,}")
        print(f"  Test: {len(test_filtered):,}")
        print(f"  Val: {len(val_filtered):,}")
        print(f"  Total: {len(train_filtered) + len(test_filtered) + len(val_filtered):,}")
        
        # Combine for analysis
        all_filtered = pd.concat([train_filtered, test_filtered, val_filtered], ignore_index=True)
        
        return train_filtered, test_filtered, val_filtered, all_filtered
    
    def create_visualizations(self, original_data, filtered_data, train_filtered, test_filtered, val_filtered):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # 1. Language Distribution Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original language distribution
        if 'lang' in original_data.columns:
            lang_counts = original_data['lang'].value_counts().head(10)
            ax1.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%')
            ax1.set_title('Original Dataset\nLanguage Distribution (Top 10)')
        
        # Filtered language distribution
        if 'lang' in filtered_data.columns:
            filtered_lang_counts = filtered_data['lang'].value_counts()
            colors = ['#1f77b4', '#ff7f0e']
            ax2.pie(filtered_lang_counts.values, labels=['Python', 'Undefined'], 
                   autopct='%1.1f%%', colors=colors)
            ax2.set_title('Filtered Dataset\nLanguage Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'language_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Class Distribution Across Splits
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        datasets = [
            (train_filtered, 'Training', axes[0, 0]),
            (test_filtered, 'Test', axes[0, 1]),
            (val_filtered, 'Validation', axes[1, 0]),
            (filtered_data, 'Combined', axes[1, 1])
        ]
        
        for df, title, ax in datasets:
            if len(df) > 0 and 'y' in df.columns:
                counts = df['y'].value_counts().sort_index()
                labels = ['No Review Needed', 'Review Needed']
                colors = ['lightgreen', 'lightcoral']
                
                bars = ax.bar(labels, counts.values, color=colors)
                ax.set_title(f'{title} Set\nClass Distribution')
                ax.set_ylabel('Count')
                
                # Add percentages on bars
                total = counts.sum()
                for bar, count in zip(bars, counts.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                           f'{count:,}\n({count/total*100:.1f}%)',
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_splits.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Feature Distributions
        self.plot_feature_distributions(filtered_data)
        
        # 4. Feature vs Label Analysis
        self.plot_feature_label_analysis(filtered_data)
        
        # 5. Correlation Matrix
        self.plot_correlation_matrix(filtered_data)
    
    def plot_feature_distributions(self, df):
        """Plot distributions of numerical features"""
        if len(df) == 0:
            return
        
        # Select numerical features
        numerical_features = ['patch_length', 'num_additions', 'num_deletions', 
                            'total_changes', 'message_length']
        
        # Filter features that exist
        available_features = [f for f in numerical_features if f in df.columns]
        
        if not available_features:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features):
            if i >= len(axes):
                break
                
            # Remove extreme outliers for better visualization
            data = df[feature].copy()
            q99 = data.quantile(0.99)
            data_clipped = data[data <= q99]
            
            axes[i].hist(data_clipped, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}')
            axes[i].set_xlabel(feature.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'Mean: {data.mean():.1f}\nMedian: {data.median():.1f}\nStd: {data.std():.1f}'
            axes[i].text(0.7, 0.7, stats_text, transform=axes[i].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_label_analysis(self, df):
        """Analyze features by label"""
        if len(df) == 0 or 'y' not in df.columns:
            return
        
        numerical_features = ['patch_length', 'num_additions', 'num_deletions', 
                            'total_changes', 'message_length']
        available_features = [f for f in numerical_features if f in df.columns]
        
        if not available_features:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features):
            if i >= len(axes):
                break
            
            # Box plot by label
            data_no_review = df[df['y'] == 0][feature]
            data_review_needed = df[df['y'] == 1][feature]
            
            # Remove extreme outliers for better visualization
            q99_no = data_no_review.quantile(0.99)
            q99_yes = data_review_needed.quantile(0.99)
            q99 = min(q99_no, q99_yes)
            
            data_plot = [
                data_no_review[data_no_review <= q99],
                data_review_needed[data_review_needed <= q99]
            ]
            
            box_plot = axes[i].boxplot(data_plot, labels=['No Review', 'Review Needed'])
            axes[i].set_title(f'{feature.replace("_", " ").title()} by Label')
            axes[i].set_ylabel(feature.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add mean values
            means = [data_no_review.mean(), data_review_needed.mean()]
            axes[i].scatter([1, 2], means, color='red', marker='x', s=100, label='Mean')
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_by_label_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix of numerical features"""
        if len(df) == 0:
            return
        
        numerical_features = ['patch_length', 'num_additions', 'num_deletions', 
                            'total_changes', 'message_length', 'y']
        available_features = [f for f in numerical_features if f in df.columns]
        
        if len(available_features) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = df[available_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_statistics(self, original_data, filtered_data, train_df, test_df, val_df):
        """Generate comprehensive summary statistics"""
        summary = {}
        
        # Original dataset stats
        summary['original'] = {
            'total_samples': len(original_data),
            'languages': original_data['lang'].nunique() if 'lang' in original_data.columns else 0,
            'top_languages': original_data['lang'].value_counts().head(10).to_dict() if 'lang' in original_data.columns else {},
            'projects': original_data['proj'].nunique() if 'proj' in original_data.columns else 0,
            'label_distribution': original_data['y'].value_counts().to_dict() if 'y' in original_data.columns else {}
        }
        
        # Filtered dataset stats
        summary['filtered'] = {
            'total_samples': len(filtered_data),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'val_samples': len(val_df),
            'language_distribution': filtered_data['lang'].value_counts().to_dict() if 'lang' in filtered_data.columns else {},
            'projects': filtered_data['proj'].nunique() if 'proj' in filtered_data.columns else 0
        }
        
        # Feature statistics for filtered data
        if len(filtered_data) > 0:
            numerical_features = ['patch_length', 'num_additions', 'num_deletions', 
                                'total_changes', 'message_length']
            available_features = [f for f in numerical_features if f in filtered_data.columns]
            
            if available_features:
                summary['filtered']['feature_stats'] = {}
                for feature in available_features:
                    summary['filtered']['feature_stats'][feature] = {
                        'mean': float(filtered_data[feature].mean()),
                        'median': float(filtered_data[feature].median()),
                        'std': float(filtered_data[feature].std()),
                        'min': float(filtered_data[feature].min()),
                        'max': float(filtered_data[feature].max())
                    }
        
        # Class distribution by split (filtered data)
        for split_name, split_df in [('train', train_df), ('test', test_df), ('val', val_df)]:
            if len(split_df) > 0 and 'y' in split_df.columns:
                label_dist = split_df['y'].value_counts()
                summary['filtered'][f'{split_name}_label_distribution'] = {
                    'no_review': int(label_dist.get(0, 0)),
                    'review_needed': int(label_dist.get(1, 0)),
                    'total': len(split_df)
                }
        
        # Save summary to JSON
        with open(self.output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def create_latex_tables(self, summary):
        """Generate LaTeX table code for the paper"""
        latex_tables = []
        
        # Table 1: Dataset Overview
        table1 = """
\\begin{table}[h]
\\centering
\\caption{Dataset Overview and Composition}
\\label{tab:dataset_overview}
\\begin{tabular}{|l|r|r|r|}
\\hline
\\textbf{Dataset} & \\textbf{Samples} & \\textbf{Languages} & \\textbf{Projects} \\\\
\\hline
"""
        
        original_samples = summary['original']['total_samples']
        original_langs = summary['original']['languages']
        original_projs = summary['original']['projects']
        
        filtered_samples = summary['filtered']['total_samples']
        filtered_projs = summary['filtered']['projects']
        
        table1 += f"Original (Full) & {original_samples:,} & {original_langs} & {original_projs:,} \\\\\n"
        table1 += f"Filtered (Py+Undefined) & {filtered_samples:,} & 2 & {filtered_projs:,} \\\\\n"
        table1 += """\\hline
\\end{tabular}
\\end{table}
"""
        latex_tables.append(table1)
        
        # Table 2: Split Distribution
        table2 = """
\\begin{table}[h]
\\centering
\\caption{Dataset Split Distribution (Filtered)}
\\label{tab:split_distribution}
\\begin{tabular}{|l|r|r|r|}
\\hline
\\textbf{Split} & \\textbf{Samples} & \\textbf{No Review (0)} & \\textbf{Review Needed (1)} \\\\
\\hline
"""
        
        for split in ['train', 'test', 'val']:
            split_data = summary['filtered'].get(f'{split}_label_distribution', {})
            total = split_data.get('total', 0)
            no_review = split_data.get('no_review', 0)
            review_needed = split_data.get('review_needed', 0)
            
            if total > 0:
                no_review_pct = (no_review / total) * 100
                review_needed_pct = (review_needed / total) * 100
                
                split_name = split.capitalize()
                table2 += f"{split_name} & {total:,} & {no_review:,} ({no_review_pct:.1f}\\%) & {review_needed:,} ({review_needed_pct:.1f}\\%) \\\\\n"
        
        table2 += """\\hline
\\end{tabular}
\\end{table}
"""
        latex_tables.append(table2)
        
        # Table 3: Feature Statistics
        if 'feature_stats' in summary['filtered']:
            table3 = """
\\begin{table}[h]
\\centering
\\caption{Numerical Feature Statistics}
\\label{tab:feature_stats}
\\begin{tabular}{|l|r|r|r|r|r|}
\\hline
\\textbf{Feature} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} \\\\
\\hline
"""
            
            feature_names = {
                'patch_length': 'Patch Length',
                'num_additions': 'Additions',
                'num_deletions': 'Deletions',
                'total_changes': 'Total Changes',
                'message_length': 'Message Length'
            }
            
            for feature, stats in summary['filtered']['feature_stats'].items():
                display_name = feature_names.get(feature, feature.replace('_', ' ').title())
                table3 += f"{display_name} & {stats['mean']:.1f} & {stats['median']:.1f} & {stats['std']:.1f} & {stats['min']:.0f} & {stats['max']:,.0f} \\\\\n"
            
            table3 += """\\hline
\\end{tabular}
\\end{table}
"""
            latex_tables.append(table3)
        
        # Save LaTeX tables
        with open(self.output_dir / 'latex_tables.tex', 'w') as f:
            f.write('\n'.join(latex_tables))
        
        print(f"LaTeX tables saved to {self.output_dir / 'latex_tables.tex'}")
        return latex_tables
    
    def run_full_analysis(self, sample_size=None):
        """Run complete dataset analysis"""
        print("Starting comprehensive dataset analysis...")
        
        # Load data
        train_df, test_df, val_df = self.load_all_data(sample_size)
        
        # Analyze original data
        original_data = self.analyze_original_data(train_df, test_df, val_df)
        
        # Analyze filtered data
        train_filtered, test_filtered, val_filtered, filtered_data = self.analyze_filtered_data(
            train_df, test_df, val_df)
        
        # Generate summary statistics
        summary = self.generate_summary_statistics(
            original_data, filtered_data, train_filtered, test_filtered, val_filtered)
        
        # Create visualizations
        self.create_visualizations(
            original_data, filtered_data, train_filtered, test_filtered, val_filtered)
        
        # Generate LaTeX tables
        latex_tables = self.create_latex_tables(summary)
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"All outputs saved to: {self.output_dir}")
        print(f"Generated files:")
        print(f"  - language_distribution_comparison.png")
        print(f"  - class_distribution_splits.png")
        print(f"  - feature_distributions.png")
        print(f"  - feature_by_label_analysis.png")
        print(f"  - correlation_matrix.png")
        print(f"  - dataset_summary.json")
        print(f"  - latex_tables.tex")
        
        return summary, latex_tables

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DatasetInsightsAnalyzer(data_dir="./data", output_dir="./dataset_insights")
    
    # Run full analysis
    # For quick testing, you can use sample_size=10000
    summary, latex_tables = analyzer.run_full_analysis(sample_size=None)
    
    # Print key insights
    print(f"\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    if 'original' in summary and 'filtered' in summary:
        original_total = summary['original']['total_samples']
        filtered_total = summary['filtered']['total_samples']
        reduction_pct = ((original_total - filtered_total) / original_total) * 100
        
        print(f"Dataset Reduction: {original_total:,} → {filtered_total:,} ({reduction_pct:.1f}% reduction)")
        print(f"Focus Languages: Python + Undefined ({filtered_total:,} samples)")
        
        if 'top_languages' in summary['original']:
            print(f"\nOriginal Top Languages:")
            for lang, count in list(summary['original']['top_languages'].items())[:5]:
                print(f"  {lang}: {count:,}")
        
        if 'train_label_distribution' in summary['filtered']:
            train_dist = summary['filtered']['train_label_distribution']
            print(f"\nTraining Set Balance:")
            print(f"  No Review: {train_dist['no_review']:,}")
            print(f"  Review Needed: {train_dist['review_needed']:,}")
            print(f"  Ratio: 1:{train_dist['review_needed']/train_dist['no_review']:.2f}")

# Additional utility functions for specific analyses

def compare_language_distributions(analyzer):
    """Compare language distributions before and after filtering"""
    print("Generating detailed language comparison...")
    
    train_df, test_df, val_df = analyzer.load_all_data()
    original_data = pd.concat([train_df, test_df, val_df], ignore_index=True)
    original_data = analyzer.clean_data(original_data)
    
    if 'lang' in original_data.columns:
        lang_dist = original_data['lang'].value_counts()
        
        print(f"\nComplete Language Distribution (Original Dataset):")
        print(f"{'Language':<15} {'Count':<10} {'Percentage':<12}")
        print("-" * 40)
        
        for lang, count in lang_dist.items():
            pct = (count / len(original_data)) * 100
            print(f"{lang:<15} {count:<10,} {pct:<12.2f}%")
        
        # Calculate how much data we're keeping
        python_count = lang_dist.get('py', 0)
        undefined_count = lang_dist.get('undefined', 0)
        kept_total = python_count + undefined_count
        
        print(f"\nFiltering Impact:")
        print(f"  Python: {python_count:,} samples")
        print(f"  Undefined: {undefined_count:,} samples")
        print(f"  Total Kept: {kept_total:,} ({kept_total/len(original_data)*100:.1f}%)")
        print(f"  Total Removed: {len(original_data) - kept_total:,} ({(len(original_data) - kept_total)/len(original_data)*100:.1f}%)")

def analyze_patch_complexity(df):
    """Analyze patch complexity patterns"""
    if len(df) == 0 or 'y' not in df.columns:
        return
    
    print(f"\nPatch Complexity Analysis:")
    
    # Define complexity categories
    df['complexity_category'] = 'Simple'
    df.loc[df['total_changes'] >= 10, 'complexity_category'] = 'Medium'
    df.loc[df['total_changes'] >= 50, 'complexity_category'] = 'Complex'
    df.loc[df['total_changes'] >= 100, 'complexity_category'] = 'Very Complex'
    
    # Analyze by complexity and label
    complexity_analysis = df.groupby(['complexity_category', 'y']).size().unstack(fill_value=0)
    
    if not complexity_analysis.empty:
        complexity_analysis['total'] = complexity_analysis.sum(axis=1)
        complexity_analysis['review_rate'] = (complexity_analysis[1] / complexity_analysis['total'] * 100).round(2)
        
        print(complexity_analysis)
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stacked bar chart
        complexity_analysis[[0, 1]].plot(kind='bar', stacked=True, ax=ax1, 
                                        color=['lightgreen', 'lightcoral'])
        ax1.set_title('Samples by Complexity and Review Need')
        ax1.set_xlabel('Complexity Category')
        ax1.set_ylabel('Number of Samples')
        ax1.legend(['No Review', 'Review Needed'])
        ax1.tick_params(axis='x', rotation=45)
        
        # Review rate by complexity
        ax2.bar(complexity_analysis.index, complexity_analysis['review_rate'], 
                color='skyblue', edgecolor='black')
        ax2.set_title('Review Rate by Complexity')
        ax2.set_xlabel('Complexity Category')
        ax2.set_ylabel('Review Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(complexity_analysis['review_rate']):
            ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(Path("./dataset_insights") / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Quick run function for testing
def quick_analysis(sample_size=50000):
    """Run a quick analysis with a sample of the data"""
    print(f"Running quick analysis with {sample_size:,} samples...")
    
    analyzer = DatasetInsightsAnalyzer()
    summary, latex_tables = analyzer.run_full_analysis(sample_size=sample_size)
    
    return summary, latex_tables