#!/usr/bin/env python3
"""
Complete analysis pipeline for code review dataset
Run this script to get comprehensive statistics and visualizations
"""

import json
import os
from pathlib import Path

# Import our custom classes (assuming they're in the same directory)
from datasetAnalyzer import CodeReviewDatasetAnalyzer
from datasetVisulaizer   import DatasetVisualizer
from additional_analysis import run_additional_analysis, export_statistics_report, generate_dataset_summary_table


def main():
    """Main analysis pipeline"""

    # Configuration
    data_paths = {
        'train': 'msg-train.jsonl',
        'test': 'msg-test.jsonl',
        'valid': 'msg-valid.jsonl'
    }

    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)

    print("🔍 Starting comprehensive dataset analysis...")
    print(f"📁 Results will be saved to: {output_dir}")

    # Check if files exist
    missing_files = []
    for split, path in data_paths.items():
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("Please ensure all dataset files are in the current directory.")
        return

    try:
        # Step 1: Basic Analysis
        print("\n📊 Step 1: Running basic statistical analysis...")
        analyzer = CodeReviewDatasetAnalyzer(data_paths)
        stats = analyzer.generate_full_report()

        # Save basic statistics
        stats_file = output_dir / 'dataset_statistics.json'
        analyzer.save_stats_to_file(stats, str(stats_file))

        # Print summary
        analyzer.print_summary_report(stats)

        # Step 2: Additional Analysis
        print("\n🔬 Step 2: Running additional analysis...")
        additional_stats = run_additional_analysis(data_paths)

        # Save additional statistics
        additional_file = output_dir / 'additional_analysis.json'
        with open(additional_file, 'w') as f:
            json.dump(additional_stats, f, indent=2)

        # Step 3: Generate Text Report
        print("\n📝 Step 3: Generating text report...")
        report_file = output_dir / 'dataset_report.txt'
        export_statistics_report(stats, str(report_file))

        # Step 4: Create Visualizations
        print("\n📈 Step 4: Creating visualizations...")
        visualizer = DatasetVisualizer(stats_dict=stats)

        # Save individual plots
        import matplotlib.pyplot as plt

        # Dataset distribution
        visualizer.plot_dataset_distribution()
        plt.savefig(output_dir / 'dataset_distribution.jpeg', dpi=300, bbox_inches='tight')
        plt.close()

        # Language distribution
        visualizer.plot_language_distribution()
        plt.savefig(output_dir / 'language_distribution.jpeg', dpi=300, bbox_inches='tight')
        plt.close()

        # Message categories
        visualizer.plot_message_categories()
        plt.savefig(output_dir / 'message_categories.jpeg', dpi=300, bbox_inches='tight')
        plt.close()

        # Quality metrics
        visualizer.plot_quality_metrics()
        plt.savefig(output_dir / 'quality_metrics.jpeg', dpi=300, bbox_inches='tight')
        plt.close()

        # Step 5: Generate Summary Table
        print("\n📋 Step 5: Creating summary table...")
        summary_df = generate_dataset_summary_table(stats)
        summary_file = output_dir / 'summary_table.csv'
        summary_df.to_csv(summary_file, index=False)

        # Step 6: Create Sample Analysis
        print("\n🔍 Step 6: Creating sample analysis...")
        sample_analysis = create_sample_analysis(data_paths, output_dir)

        # Final Summary
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📁 All results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  📊 {stats_file.name} - Complete statistics")
        print(f"  📋 {additional_file.name} - Additional analysis")
        print(f"  📝 {report_file.name} - Text report")
        print(f"  📈 *.png - Visualization plots")
        print(f"  📊 {summary_file.name} - Summary table")
        print(f"  🔍 sample_analysis.json - Sample data analysis")

        # Print key insights
        print_key_insights(stats, additional_stats)

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def create_sample_analysis(data_paths, output_dir):
    """Analyze a sample of the data for manual inspection"""

    # Load a sample from each split
    samples = {}

    for split, path in data_paths.items():
        with open(path, 'r') as f:
            lines = f.readlines()
            # Take first 5 samples from each split
            sample_data = []
            for i, line in enumerate(lines[:5]):
                item = json.loads(line.strip())
                sample_data.append({
                    'id': item.get('id'),
                    'message_length': len(item.get('msg', '')),
                    'message_preview': item.get('msg', '')[:100] + '...' if len(item.get('msg', '')) > 100 else item.get('msg', ''),
                    'language': item.get('lang', 'unknown'),
                    'patch_lines': item.get('patch', '').count('\n'),
                    'file_lines': item.get('oldf', '').count('\n')
                })
            samples[split] = sample_data

    # Save sample analysis
    sample_file = output_dir / 'sample_analysis.json'
    with open(sample_file, 'w') as f:
        json.dump(samples, f, indent=2)

    return samples

def print_key_insights(stats, additional_stats):
    """Print key insights from the analysis"""

    print("\n🔑 KEY INSIGHTS")
    print("-" * 30)

    basic = stats['basic_statistics']
    text = stats['text_statistics']
    code = stats['code_statistics']
    quality = stats['quality_metrics']
    content = stats['content_analysis']

    # Dataset characteristics
    print(f"💾 Dataset has {basic['total_samples']:,} total samples")

    # Most common language
    lang_dist = code['language_distribution']
    if lang_dist:
        top_lang = max(lang_dist.items(), key=lambda x: x[1])
        print(f"🏆 Most common language: {top_lang[0]} ({top_lang[1]} files)")

    # Message characteristics
    avg_length = text['message_stats']['avg_length']
    if avg_length < 50:
        print(f"📝 Messages are quite short (avg: {avg_length:.1f} chars) - might be brief feedback")
    elif avg_length > 200:
        print(f"📝 Messages are detailed (avg: {avg_length:.1f} chars) - comprehensive reviews")
    else:
        print(f"📝 Messages have moderate length (avg: {avg_length:.1f} chars)")

    # Quality assessment
    if quality['completeness_rate'] > 95:
        print(f"✅ High data quality: {quality['completeness_rate']:.1f}% complete")
    elif quality['completeness_rate'] > 80:
        print(f"⚠️  Good data quality: {quality['completeness_rate']:.1f}% complete")
    else:
        print(f"❌ Data quality concerns: only {quality['completeness_rate']:.1f}% complete")

    # Most common review category
    categories = content['message_categories']
    if categories:
        top_category = max(categories.items(), key=lambda x: x[1])
        print(f"🎯 Most common review type: {top_category[0]} ({top_category[1]} messages)")

    # Patch complexity
    if 'patch_analysis' in code:
        avg_lines_added = code['patch_analysis']['lines_added']['mean']
        if avg_lines_added < 5:
            print(f"🔧 Small changes: avg {avg_lines_added:.1f} lines added per patch")
        elif avg_lines_added > 20:
            print(f"🔧 Large changes: avg {avg_lines_added:.1f} lines added per patch")
        else:
            print(f"🔧 Moderate changes: avg {avg_lines_added:.1f} lines added per patch")

    # Sentiment
    sentiment = content['sentiment_analysis']
    total_sentiment = sum(sentiment.values())
    if total_sentiment > 0:
        positive_pct = sentiment['positive'] / total_sentiment * 100
        negative_pct = sentiment['negative'] / total_sentiment * 100
        if positive_pct > 50:
            print(f"😊 Generally positive reviews ({positive_pct:.1f}% positive)")
        elif negative_pct > 50:
            print(f"😟 Generally critical reviews ({negative_pct:.1f}% negative)")
        else:
            print(f"😐 Balanced sentiment in reviews")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()