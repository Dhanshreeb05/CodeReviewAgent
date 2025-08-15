# Run this script to generate dataset insights
# Make sure you have the required packages installed:
# pip install pandas matplotlib seaborn numpy

from dataset_insights_analyzer import DatasetInsightsAnalyzer, compare_language_distributions, analyze_patch_complexity

def main():
    print("=" * 80)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = DatasetInsightsAnalyzer(data_dir="./data", output_dir="./dataset_insights")
    
    # Option 1: Quick analysis with sample (faster for testing)
    # summary, latex_tables = analyzer.run_full_analysis(sample_size=50000)
    
    # Option 2: Full analysis (slower but complete)
    print("Running full dataset analysis...")
    summary, latex_tables = analyzer.run_full_analysis(sample_size=None)
    
    # Additional detailed analyses
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSES")
    print("=" * 60)
    
    # Language comparison
    compare_language_distributions(analyzer)
    
    # Load filtered data for complexity analysis
    train_df, test_df, val_df = analyzer.load_all_data()
    train_filtered = analyzer.filter_python_undefined(analyzer.clean_data(train_df))
    train_filtered = analyzer.extract_features(train_filtered)
    
    # Patch complexity analysis
    analyze_patch_complexity(train_filtered)
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Check the './dataset_insights/' folder for:")
    print(f"  📊 Visualizations (PNG files)")
    print(f"  📋 LaTeX tables (latex_tables.tex)")
    print(f"  📈 Summary statistics (dataset_summary.json)")
    
    # Display LaTeX tables for easy copy-paste
    print(f"\n" + "=" * 60)
    print("LATEX TABLES FOR YOUR PAPER")
    print("=" * 60)
    
    for i, table in enumerate(latex_tables, 1):
        print(f"\n--- Table {i} ---")
        print(table)

if __name__ == "__main__":
    main()