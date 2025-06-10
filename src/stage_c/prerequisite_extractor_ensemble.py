import os
import json
from typing import Dict, List
import joblib
import numpy as np
import pickle

# Import the ensemble model implementation
from train_ranker_ensemble import EnsemblePrerequisiteRanker

class EnsembleRanker:
    """
    Ranks expressions using an ensemble of feature-based and encoder-based models.
    """
    
    def __init__(self, ensemble_dir: str, feature_model_path: str, encoder_model_path: str):
        """
        Initialize the ensemble ranker.
        
        Args:
            ensemble_dir: Directory containing the ensemble model
            feature_model_path: Path to the feature-based model
            encoder_model_path: Path to the encoder-based model
        """
        try:
            self.ensemble = EnsemblePrerequisiteRanker.load(
                model_dir=ensemble_dir,
                feature_model_path=feature_model_path,
                encoder_model_path=encoder_model_path
            )
            print(f"Ensemble ranker loaded from {ensemble_dir}")
        except Exception as e:
            raise RuntimeError(f"Error loading ensemble model: {e}")
    
    def rank_expressions(self, 
                         filtered_expressions: Dict[str, float], 
                         document_text: str) -> Dict[str, int]:
        """
        Rank filtered expressions using the ensemble model.
        
        Args:
            filtered_expressions: Dictionary of expressions to similarity scores
            document_text: Full text of the document
            
        Returns:
            Dictionary mapping expressions to their predicted ranks (0-3)
        """
        if not filtered_expressions:
            return {}
            
        expressions = list(filtered_expressions.keys())
        
        try:
            # Use the ensemble model to predict ranks
            ranked_expressions = self.ensemble.predict(
                document_text=document_text,
                expressions=expressions,
                similarity_scores=filtered_expressions
            )
            
            return ranked_expressions
            
        except Exception as e:
            print(f"Error during ensemble prediction: {e}")
            # Fallback: return expressions with default rank 0
            return {expr: 0 for expr in expressions}


def save_ranked_expressions(ranked_expressions: Dict[str, int], output_path: str):
    """
    Save ranked expressions to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranked_expressions, f, ensure_ascii=False, indent=4)
    print(f"Ranked prerequisites saved to {output_path}")


def process_page_ranking(page_title: str,
                         raw_data_dir: str,
                         stage_b_output_dir: str,
                         stage_c_output_dir: str,
                         ranker: EnsembleRanker) -> Dict[str, int]:
    """
    Process a single page: load raw text and Stage B results, rank expressions, save results.
    
    Args:
        page_title: Title of the page
        raw_data_dir: Directory containing raw text files
        stage_b_output_dir: Directory with Stage B outputs
        stage_c_output_dir: Directory to save ranked expressions
        ranker: Ensemble ranker instance
    """
    # Load raw document text
    raw_file_path = os.path.join(raw_data_dir, f"{page_title}.txt")
    if not os.path.exists(raw_file_path):
        print(f"Warning: Raw text file not found for {page_title}: {raw_file_path}")
        return {}
        
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Load filtered expressions from Stage B
    stage_b_file_path = os.path.join(stage_b_output_dir, f"{page_title}_filtered.json")
    if not os.path.exists(stage_b_file_path):
        print(f"Warning: Stage B output file not found for {page_title}: {stage_b_file_path}")
        return {}
        
    with open(stage_b_file_path, 'r', encoding='utf-8') as f:
        filtered_expressions = json.load(f)

    # Rank expressions using the ensemble model
    ranked_expressions = ranker.rank_expressions(filtered_expressions, document_text)

    # Save results
    output_path = os.path.join(stage_c_output_dir, f"{page_title}_ensemble_ranked_prerequisites.json")
    save_ranked_expressions(ranked_expressions, output_path)

    return ranked_expressions


def main():
    """
    Main function to process all pages using the ensemble ranker.
    """
    # Configuration
    raw_data_dir = "data/raw"
    stage_b_output_dir = "data/processed/stage_b" 
    stage_c_output_dir = "data/processed/stage_c/ensemble"
    
    # Model paths
    ensemble_dir = "models/ensemble_ranker"
    feature_model_path = "models/stage_c_ranker.joblib"
    encoder_model_path = "models/encoder_ranker.pt"
    
    # Check if models exist
    if not os.path.exists(ensemble_dir):
        print(f"Error: Ensemble model directory not found at {ensemble_dir}")
        print("Please run train_ranker_ensemble.py first.")
        return
        
    if not os.path.exists(feature_model_path):
        print(f"Error: Feature model not found at {feature_model_path}")
        print("Please run train_ranker.py first.")
        return
        
    if not os.path.exists(encoder_model_path):
        print(f"Error: Encoder model not found at {encoder_model_path}")
        print("Please run train_ranker_encoder.py first.")
        return

    # Initialize the ensemble ranker
    try:
        ranker = EnsembleRanker(
            ensemble_dir=ensemble_dir,
            feature_model_path=feature_model_path,
            encoder_model_path=encoder_model_path
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Failed to initialize ensemble ranker: {e}")
        return

    # Ensure output directory exists
    os.makedirs(stage_c_output_dir, exist_ok=True)

    # Get all page titles from Stage B output
    if not os.path.exists(stage_b_output_dir):
        print(f"Error: Stage B output directory not found: {stage_b_output_dir}")
        return

    page_titles = set()
    for filename in os.listdir(stage_b_output_dir):
        if filename.endswith("_filtered.json"):
            page_titles.add(filename.replace("_filtered.json", ""))
    
    if not page_titles:
        print(f"No processed pages found in {stage_b_output_dir}")
        return

    # Process each page
    for page_title in sorted(list(page_titles)):
        print(f"Ranking prerequisites for page: {page_title}")
        ranked_prerequisites = process_page_ranking(
            page_title=page_title,
            raw_data_dir=raw_data_dir,
            stage_b_output_dir=stage_b_output_dir,
            stage_c_output_dir=stage_c_output_dir,
            ranker=ranker
        )
        print(f"Ranked {len(ranked_prerequisites)} expressions for page {page_title}")


if __name__ == "__main__":
    main()
