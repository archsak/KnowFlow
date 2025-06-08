import os
import json
from typing import List, Dict, Tuple
import joblib # Used for loading scikit-learn style models
import numpy as np # For preparing model input

class PrerequisiteRanker:
    """
    Ranks expressions based on their importance for understanding a text
    using a pre-trained supervised model.
    Assigns an importance score from 0 to 3.
    """

    def __init__(self, model_path: str):
        """
        Initialize the ranker with a path to a trained supervised model.

        Args:
            model_path: Path to the trained ranking model file (e.g., a .joblib file).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ranker model not found at {model_path}")
        try:
            self.model = joblib.load(model_path)
            print(f"Prerequisite ranking model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")

    def rank_expressions(self, filtered_expressions: Dict[str, float]) -> Dict[str, int]:
        """
        Rank filtered expressions by predicting an importance score using the loaded model.

        Args:
            filtered_expressions: Dictionary mapping expressions to their similarity scores (from Stage B).
                                  The similarity score is assumed to be a feature for the model.

        Returns:
            Dictionary mapping expressions to their predicted importance scores (0-3).
        """
        ranked_expressions = {}
        if not filtered_expressions:
            return {}

        # Prepare features for batch prediction if model supports it, or predict one by one
        # Assuming the model expects a 2D array of features,
        # where each row is an expression and columns are features.
        # For now, we assume the primary feature is the similarity score.
        
        expressions_list = list(filtered_expressions.keys())
        similarity_scores = np.array([filtered_expressions[expr] for expr in expressions_list]).reshape(-1, 1)

        try:
            # Predict ranks using the loaded model
            # The shape of `similarity_scores` should match what the model expects.
            # If your model expects other features, this part needs to be adjusted.
            predicted_ranks = self.model.predict(similarity_scores)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Fallback or error handling: assign a default rank or re-raise
            # For now, let's assign a default rank of 0 or skip if prediction fails.
            # This depends on desired behavior.
            return {expr: 0 for expr in expressions_list}


        for i, expr in enumerate(expressions_list):
            # Ensure rank is an integer and within expected range (0-3)
            rank = int(round(predicted_ranks[i]))
            rank = max(0, min(3, rank)) # Clamp to 0-3 range
            ranked_expressions[expr] = rank
        
        # Sort by predicted importance score (descending),
        # then by original similarity (descending) as a tie-breaker.
        sorted_ranked_expressions = dict(sorted(
            ranked_expressions.items(),
            key=lambda item: (item[1], filtered_expressions[item[0]]), # item[1] is rank, filtered_expressions[item[0]] is similarity
            reverse=True
        ))
        return sorted_ranked_expressions

def save_ranked_expressions(ranked_expressions: Dict[str, int], output_path: str):
    """
    Save ranked expressions to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ranked_expressions, f, ensure_ascii=False, indent=4)
    print(f"Ranked prerequisites saved to {output_path}")

def process_page_ranking(page_title: str,
                         stage_b_output_dir: str,
                         stage_c_output_dir: str,
                         ranker: PrerequisiteRanker) -> Dict[str, int]:
    """
    Process a single page: load Stage B results, rank expressions, and save Stage C results.
    """
    # Load filtered expressions from Stage B
    stage_b_file_path = os.path.join(stage_b_output_dir, f"{page_title}_filtered.json")
    if not os.path.exists(stage_b_file_path):
        print(f"Warning: Stage B output file not found for {page_title}: {stage_b_file_path}")
        return {}
        
    with open(stage_b_file_path, 'r', encoding='utf-8') as f:
        filtered_expressions = json.load(f)

    # Rank expressions
    ranked_expressions = ranker.rank_expressions(filtered_expressions)

    # Save results
    output_path = os.path.join(stage_c_output_dir, f"{page_title}_ranked_prerequisites.json")
    save_ranked_expressions(ranked_expressions, output_path)

    return ranked_expressions

def main():
    """
    Main function to process all pages for prerequisite ranking.
    """
    # Configuration
    stage_b_output_dir = "data/processed/stage_b" 
    stage_c_output_dir = "data/processed/stage_c"
    
    # Path to the trained ranker model produced by src/stage_c/train_ranker.py
    # This is a placeholder path; update it to where your model is saved.
    ranker_model_path = "models/stage_c_ranker.joblib" 

    if not os.path.exists(ranker_model_path):
        print(f"Error: Ranker model not found at {ranker_model_path}.")
        print("Please ensure `src/stage_c/train_ranker.py` has been run and the model is saved correctly.")
        return

    # Initialize the prerequisite ranker
    try:
        ranker = PrerequisiteRanker(model_path=ranker_model_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Failed to initialize PrerequisiteRanker: {e}")
        return

    # Ensure Stage C output directory exists
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
        ranked_prerequisites = process_page_ranking(page_title,
                                                    stage_b_output_dir,
                                                    stage_c_output_dir,
                                                    ranker)
        print(f"Ranked {len(ranked_prerequisites)} expressions for page {page_title}")

if __name__ == "__main__":
    main()
