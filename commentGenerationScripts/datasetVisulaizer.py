import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import json


class DatasetVisualizer:
    def __init__(self, stats_file_path=None, stats_dict=None):
        """
        Initialize visualizer with either a stats file path or stats dictionary
        """
        if stats_file_path:
            with open(stats_file_path, 'r') as f:
                self.stats = json.load(f)
        elif stats_dict:
            self.stats = stats_dict
        else:
            raise ValueError("Either stats_file_path or stats_dict must be provided")

    def plot_dataset_distribution(self):
        """Plot dataset split distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart for split distribution
        splits = self.stats['basic_statistics']['dataset_sizes']
        ax1.pie(splits.values(), labels=splits.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title('Dataset Split Distribution')

        # Bar chart for absolute numbers
        ax2.bar(splits.keys(), splits.values())
        ax2.set_title('Dataset Split Sizes')
        ax2.set_ylabel('Number of Samples')

        # Add value labels on bars
        for i, v in enumerate(splits.values()):
            ax2.text(i, v + max(splits.values()) * 0.01, f'{v:,}', ha='center')

        plt.tight_layout()
        plt.show()

    def plot_language_distribution(self, top_n=10):
        """Plot programming language distribution"""
        lang_dist = self.stats['code_statistics']['language_distribution']

        # Get top N languages
        sorted_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
        languages, counts = zip(*sorted_langs)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(languages, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title(f'Top {top_n} Programming Languages')
        plt.xlabel('Programming Language')
        plt.ylabel('Number of Files')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_message_length_distribution(self):
        """Plot message length statistics"""
        msg_stats = self.stats['text_statistics']['message_stats']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Statistics summary
        stats_text = f"""
        Average Length: {msg_stats['avg_length']:.1f} chars
        Median Length: {msg_stats['median_length']:.1f} chars
        Min Length: {msg_stats['min_length']} chars
        Max Length: {msg_stats['max_length']} chars
        Std Dev: {msg_stats['std_length']:.1f} chars

        Average Words: {msg_stats['avg_word_count']:.1f}
        Empty Messages: {msg_stats['empty_messages']}
        """

        ax1.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Message Length Statistics')

        # Create sample data for histogram (you'd need to modify this to use actual data)
        # This is a placeholder - you'd want to pass actual message lengths
        sample_lengths = np.random.normal(msg_stats['avg_length'], msg_stats['std_length'], 1000)
        sample_lengths = np.clip(sample_lengths, 0, None)  # No negative lengths

        ax2.hist(sample_lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Message Length Distribution')
        ax2.set_xlabel('Message Length (characters)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(msg_stats['avg_length'], color='red', linestyle='--', label='Mean')
        ax2.axvline(msg_stats['median_length'], color='orange', linestyle='--', label='Median')
        ax2.legend()

        # Box plot
        ax3.boxplot([sample_lengths], labels=['Message Length'])
        ax3.set_title('Message Length Box Plot')
        ax3.set_ylabel('Length (characters)')

        # Word count distribution (placeholder)
        sample_words = np.random.normal(msg_stats['avg_word_count'], msg_stats['avg_word_count'] / 3, 1000)
        sample_words = np.clip(sample_words, 1, None)

        ax4.hist(sample_words, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_title('Word Count Distribution')
        ax4.set_xlabel('Number of Words')
        ax4.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_patch_analysis(self):
        """Plot patch analysis statistics"""
        patch_stats = self.stats['code_statistics']['patch_analysis']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Lines added/removed comparison
        metrics = ['lines_added', 'lines_removed']
        means = [patch_stats[metric]['mean'] for metric in metrics]

        ax1.bar(metrics, means, color=['green', 'red'], alpha=0.7)
        ax1.set_title('Average Lines Added vs Removed')
        ax1.set_ylabel('Average Lines')

        # Add value labels
        for i, v in enumerate(means):
            ax1.text(i, v + max(means) * 0.01, f'{v:.1f}', ha='center')

        # Files changed distribution
        files_stats = patch_stats['files_changed']
        ax2.bar(['Mean', 'Median', 'Max'],
                [files_stats['mean'], files_stats['median'], files_stats['max']],
                color='skyblue', alpha=0.7)
        ax2.set_title('Files Changed per Patch')
        ax2.set_ylabel('Number of Files')

        # Hunks per patch
        hunks_stats = patch_stats['hunks_per_patch']
        ax3.bar(['Mean', 'Median', 'Max'],
                [hunks_stats['mean'], hunks_stats['median'], hunks_stats['max']],
                color='orange', alpha=0.7)
        ax3.set_title('Hunks per Patch')
        ax3.set_ylabel('Number of Hunks')

        # Summary table
        summary_data = []
        for metric, values in patch_stats.items():
            summary_data.append([
                metric.replace('_', ' ').title(),
                f"{values['mean']:.2f}",
                f"{values['median']:.2f}",
                f"{values['max']:.0f}",
                f"{values['std']:.2f}"
            ])

        table = ax4.table(cellText=summary_data,
                          colLabels=['Metric', 'Mean', 'Median', 'Max', 'Std'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax4.axis('off')
        ax4.set_title('Patch Statistics Summary')

        plt.tight_layout()
        plt.show()

    def plot_message_categories(self):
        """Plot message category distribution"""
        categories = self.stats['content_analysis']['message_categories']

        plt.figure(figsize=(12, 8))

        # Sort categories by count
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        cats, counts = zip(*sorted_categories)

        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))

        bars = plt.bar(cats, counts, color=colors, alpha=0.8, edgecolor='black')
        plt.title('Distribution of Review Message Categories')
        plt.xlabel('Category')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_sentiment_analysis(self):
        """Plot sentiment analysis results"""
        sentiment = self.stats['content_analysis']['sentiment_analysis']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        labels = list(sentiment.keys())
        sizes = list(sentiment.values())
        colors = ['lightgreen', 'lightcoral', 'lightgray']

        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Sentiment Distribution in Review Messages')

        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Sentiment Analysis Results')
        ax2.set_ylabel('Number of Messages')

        # Add value labels
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.01,
                     f'{size}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_quality_metrics(self):
        """Plot data quality metrics"""
        quality = self.stats['quality_metrics']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Completeness rate
        complete = quality['complete_samples']
        total = quality['total_samples']
        incomplete = total - complete

        ax1.pie([complete, incomplete], labels=['Complete', 'Incomplete'],
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax1.set_title(f'Data Completeness\n({quality["completeness_rate"]:.1f}% complete)')

        # Missing fields
        if quality['missing_fields']:
            fields = list(quality['missing_fields'].keys())
            counts = list(quality['missing_fields'].values())

            ax2.bar(fields, counts, color='orange', alpha=0.7)
            ax2.set_title('Missing Fields')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Missing Fields!', ha='center', va='center',
                     fontsize=16, color='green', weight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')

        # Empty fields
        if quality['empty_fields']:
            fields = list(quality['empty_fields'].keys())
            counts = list(quality['empty_fields'].values())

            ax3.bar(fields, counts, color='red', alpha=0.7)
            ax3.set_title('Empty Fields')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Empty Fields!', ha='center', va='center',
                     fontsize=16, color='green', weight='bold')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')

        # Duplicates summary
        dup_data = [
            ['Duplicate IDs', quality['duplicate_ids']],
            ['Duplicate Messages', quality['duplicate_messages']],
            ['Total Samples', quality['total_samples']],
            ['Complete Samples', quality['complete_samples']]
        ]

        table = ax4.table(cellText=dup_data,
                          colLabels=['Metric', 'Count'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax4.axis('off')
        ax4.set_title('Quality Summary')

        plt.tight_layout()
        plt.show()

    def generate_word_cloud(self, analyzer_instance):
        """Generate word cloud from review messages"""
        # This requires access to the actual message data
        # You'll need to pass the analyzer instance or load the data separately
        all_data = []
        for data in analyzer_instance.datasets.values():
            all_data.extend(data)

        # Combine all messages
        all_messages = ' '.join([item.get('msg', '') for item in all_data if item.get('msg')])

        # Create word cloud
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white',
                              max_words=100,
                              colormap='viridis').generate(all_messages)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Review Messages', fontsize=16)
        plt.tight_layout()
        plt.show()

    def create_comprehensive_dashboard(self):
        """Create a comprehensive visualization dashboard"""
        print("Creating comprehensive dataset visualization dashboard...")

        # Set style
        plt.style.use('seaborn-v0_8')

        # Generate all plots
        self.plot_dataset_distribution()
        self.plot_language_distribution()
        self.plot_message_length_distribution()
        self.plot_patch_analysis()
        self.plot_message_categories()
        self.plot_sentiment_analysis()
        self.plot_quality_metrics()

        print("Dashboard generation complete!")


# Additional utility functions
def compare_datasets(stats_list, dataset_names):
    """Compare statistics across multiple datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Compare dataset sizes
    sizes = [stats['basic_statistics']['total_samples'] for stats in stats_list]
    axes[0].bar(dataset_names, sizes, alpha=0.7)
    axes[0].set_title('Dataset Sizes Comparison')
    axes[0].set_ylabel('Number of Samples')

    # Compare average message lengths
    msg_lengths = [stats['text_statistics']['message_stats']['avg_length'] for stats in stats_list]
    axes[1].bar(dataset_names, msg_lengths, alpha=0.7, color='orange')
    axes[1].set_title('Average Message Length Comparison')
    axes[1].set_ylabel('Characters')

    # Compare unique languages
    unique_langs = [stats['code_statistics']['unique_languages'] for stats in stats_list]
    axes[2].bar(dataset_names, unique_langs, alpha=0.7, color='green')
    axes[2].set_title('Unique Languages Comparison')
    axes[2].set_ylabel('Number of Languages')

    # Compare completeness rates
    completeness = [stats['quality_metrics']['completeness_rate'] for stats in stats_list]
    axes[3].bar(dataset_names, completeness, alpha=0.7, color='red')
    axes[3].set_title('Data Completeness Comparison')
    axes[3].set_ylabel('Completeness %')

    plt.tight_layout()
    plt.show()


# Example usage for visualization
if __name__ == "__main__":
    # After running the main analyzer, you can create visualizations:

    # Option 1: Load from saved stats file
    # visualizer = DatasetVisualizer(stats_file_path='detailed_dataset_stats.json')

    # Option 2: Use stats from analyzer directly
    # analyzer = CodeReviewDatasetAnalyzer(data_paths)
    # stats = analyzer.generate_full_report()
    # visualizer = DatasetVisualizer(stats_dict=stats)

    # Create comprehensive dashboard
    # visualizer.create_comprehensive_dashboard()

    # Or create individual plots
    # visualizer.plot_dataset_distribution()
    # visualizer.plot_language_distribution()
    # etc.

    pass