import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from typing import Dict, List, Tuple

def load_results(
    feature_based_dir: str = "data/processed/stage_c",
    encoder_based_dir: str = "data/processed/stage_c/encoder"
) -> Dict[str, Tuple[Dict, Dict]]:
    """
    Load results from both feature-based and encoder-based models for comparison
    
    Args:
        feature_based_dir: Directory containing feature-based model results
        encoder_based_dir: Directory containing encoder-based model results
        
    Returns:
        Dictionary mapping page titles to tuples of (feature_results, encoder_results)
    """
    results = {}
    
    # Get list of all processed pages
    feature_files = {f.replace("_ranked_prerequisites.json", "") 
                    for f in os.listdir(feature_based_dir)
                    if f.endswith("_ranked_prerequisites.json")}
    
    encoder_files = {f.replace("_encoder_ranked_prerequisites.json", "") 
                    for f in os.listdir(encoder_based_dir)
                    if f.endswith("_encoder_ranked_prerequisites.json")}
    
    # Find pages processed by both models
    common_pages = feature_files.intersection(encoder_files)
    
    if not common_pages:
        print("No common pages found between the two models.")
        return results
    
    print(f"Found {len(common_pages)} pages processed by both models.")
    
    # Load results for each page
    for page in common_pages:
        feature_file = os.path.join(feature_based_dir, f"{page}_ranked_prerequisites.json")
        encoder_file = os.path.join(encoder_based_dir, f"{page}_encoder_ranked_prerequisites.json")
        
        if os.path.exists(feature_file) and os.path.exists(encoder_file):
            try:
                with open(feature_file, 'r', encoding='utf-8') as f:
                    feature_results = json.load(f)
                    
                with open(encoder_file, 'r', encoding='utf-8') as f:
                    encoder_results = json.load(f)
                    
                # Only include pages where both models have results
                if feature_results and encoder_results:
                    results[page] = (feature_results, encoder_results)
            except Exception as e:
                print(f"Error loading results for page {page}: {e}")
    
    print(f"Successfully loaded results for {len(results)} pages.")
    return results

def compare_rankings(results: Dict[str, Tuple[Dict, Dict]]) -> pd.DataFrame:
    """
    Compare rankings from feature-based and encoder-based models
    
    Args:
        results: Dictionary mapping page titles to (feature_results, encoder_results)
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for page, (feature_results, encoder_results) in results.items():
        # Find common expressions
        common_expressions = set(feature_results.keys()) & set(encoder_results.keys())
        
        if not common_expressions:
            print(f"No common expressions found for page {page}")
            continue
            
        # Extract rankings for common expressions
        feature_ranks = [feature_results[expr] for expr in common_expressions]
        encoder_ranks = [encoder_results[expr] for expr in common_expressions]
        
        # Calculate metrics
        agreement = sum(f == e for f, e in zip(feature_ranks, encoder_ranks)) / len(common_expressions)
        kappa = cohen_kappa_score(feature_ranks, encoder_ranks)
        
        # Create confusion matrix
        cm = confusion_matrix(
            feature_ranks, 
            encoder_ranks, 
            labels=[0, 1, 2, 3]
        )
        
        # Store metrics
        comparison_data.append({
            'page': page,
            'num_expressions': len(common_expressions),
            'agreement': agreement,
            'kappa': kappa,
            'confusion_matrix': cm
        })
    
    return pd.DataFrame(comparison_data)

def visualize_comparison(comparison_df: pd.DataFrame):
    """
    Visualize comparison results between the two models
    
    Args:
        comparison_df: DataFrame with comparison metrics
    """
    # Create output directory
    os.makedirs("results/comparison", exist_ok=True)
    
    # 1. Overall agreement metrics
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['page'], comparison_df['agreement'], color='skyblue')
    plt.xlabel('Page')
    plt.ylabel('Agreement Ratio')
    plt.title('Agreement Between Feature-based and Encoder-based Models')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('results/comparison/agreement.png')
    
    # 2. Kappa statistics
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['page'], comparison_df['kappa'], color='lightgreen')
    plt.xlabel('Page')
    plt.ylabel('Cohen\'s Kappa')
    plt.title('Cohen\'s Kappa Between Feature-based and Encoder-based Models')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('results/comparison/kappa.png')
    
    # 3. Average confusion matrix
    combined_cm = np.zeros((4, 4))
    for cm in comparison_df['confusion_matrix']:
        combined_cm += cm
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        combined_cm, 
        annot=True, 
        fmt='g', 
        cmap='Blues',
        xticklabels=['0', '1', '2', '3'],
        yticklabels=['0', '1', '2', '3']
    )
    plt.xlabel('Encoder-based Rank')
    plt.ylabel('Feature-based Rank')
    plt.title('Combined Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/comparison/confusion_matrix.png')
    
    # Save metrics to CSV
    comparison_df[['page', 'num_expressions', 'agreement', 'kappa']].to_csv(
        'results/comparison/metrics.csv',
        index=False
    )
    
    print("Comparison visualizations saved to 'results/comparison/'")
    
def analyze_ranking_differences(results: Dict[str, Tuple[Dict, Dict]]):
    """
    Analyze the specifics of different rankings between the two models
    
    Args:
        results: Dictionary mapping page titles to (feature_results, encoder_results)
    """
    all_differences = []
    
    for page, (feature_results, encoder_results) in results.items():
        # Find common expressions
        common_expressions = set(feature_results.keys()) & set(encoder_results.keys())
        
        for expr in common_expressions:
            feature_rank = feature_results[expr]
            encoder_rank = encoder_results[expr]
            
            if feature_rank != encoder_rank:
                all_differences.append({
                    'page': page,
                    'expression': expr,
                    'feature_rank': feature_rank,
                    'encoder_rank': encoder_rank,
                    'rank_difference': encoder_rank - feature_rank
                })
    
    # Convert to DataFrame for analysis
    diff_df = pd.DataFrame(all_differences)
    
    if diff_df.empty:
        print("No ranking differences found between the models.")
        return
        
    # Save full differences to CSV
    os.makedirs("results/comparison", exist_ok=True)
    diff_df.to_csv('results/comparison/rank_differences.csv', index=False)
    
    # Summarize differences
    diff_counts = diff_df.groupby(['feature_rank', 'encoder_rank']).size().reset_index(name='count')
    
    # Calculate statistics
    print("\nRanking Difference Analysis:")
    print(f"Total expressions with different ranks: {len(diff_df)}")
    print(f"Average absolute difference: {diff_df['rank_difference'].abs().mean():.2f}")
    
    # Most common differences
    print("\nMost common rank differences:")
    sorted_diffs = diff_counts.sort_values('count', ascending=False).head(5)
    for _, row in sorted_diffs.iterrows():
        print(f"Feature rank {row['feature_rank']} → Encoder rank {row['encoder_rank']}: {row['count']} instances")

def main():
    """Main function to compare the two ranking models"""
    feature_dir = "data/processed/stage_c"
    encoder_dir = "data/processed/stage_c/encoder"
    
    print("Starting model comparison...")
    
    # Load results
    results = load_results(feature_dir, encoder_dir)
    
    if not results:
        print("No results to compare. Ensure both models have been run.")
        return
    
    # Compare rankings
    comparison_df = compare_rankings(results)
    
    # Visualize comparisons
    visualize_comparison(comparison_df)
    
    # Analyze differences
    analyze_ranking_differences(results)
    
    # Print overall statistics
    print("\nOverall Comparison Metrics:")
    print(f"Average agreement: {comparison_df['agreement'].mean():.4f}")
    print(f"Average kappa: {comparison_df['kappa'].mean():.4f}")
    print("\nComparison complete! Results saved to 'results/comparison/'")

if __name__ == "__main__":
    main()
