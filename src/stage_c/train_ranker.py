import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example classifier
from sklearn.metrics import classification_report
import joblib

def load_training_data(ranked_pages_dir: str, stage_b_output_dir: str) -> pd.DataFrame:
    """
    Loads human-ranked expressions and their corresponding similarity scores from Stage B.

    Args:
        ranked_pages_dir: Directory containing CSV files with human-ranked expressions
                          (columns: 'expression', 'rank'). File names are page titles.
        stage_b_output_dir: Directory containing Stage B output JSON files
                            (e.g., '{page_title}_filtered.json') with similarity scores.

    Returns:
        A pandas DataFrame with 'similarity_score' and 'human_rank' columns.
    """
    all_training_samples = []

    if not os.path.exists(ranked_pages_dir):
        print(f"Error: Ranked pages directory not found: {ranked_pages_dir}")
        return pd.DataFrame()
    if not os.path.exists(stage_b_output_dir):
        print(f"Error: Stage B output directory not found: {stage_b_output_dir}")
        return pd.DataFrame()

    for ranked_file_name in os.listdir(ranked_pages_dir):
        if not ranked_file_name.endswith(".csv"):
            continue
        
        page_title = ranked_file_name.replace(".csv", "")
        ranked_file_path = os.path.join(ranked_pages_dir, ranked_file_name)
        stage_b_file_path = os.path.join(stage_b_output_dir, f"{page_title}_filtered.json")

        if not os.path.exists(stage_b_file_path):
            print(f"Warning: Stage B output for {page_title} not found at {stage_b_file_path}. Skipping.")
            continue

        try:
            # Load human-ranked data
            human_ranks_df = pd.read_csv(ranked_file_path)
            if not ({'expression', 'rank'}.issubset(human_ranks_df.columns)):
                print(f"Warning: CSV file {ranked_file_name} is missing 'expression' or 'rank' column. Skipping.")
                continue
            
            # Load Stage B similarity scores
            with open(stage_b_file_path, 'r', encoding='utf-8') as f:
                stage_b_scores_dict = json.load(f) # {expression: similarity_score}
            
            if not stage_b_scores_dict:
                print(f"Warning: Stage B scores for {page_title} are empty. Skipping.")
                continue

            # Merge data
            for _, row in human_ranks_df.iterrows():
                expression = row['expression']
                human_rank = row['rank']
                
                if expression in stage_b_scores_dict:
                    similarity_score = stage_b_scores_dict[expression]
                    all_training_samples.append({
                        'similarity_score': similarity_score,
                        'human_rank': int(human_rank) # Ensure rank is integer
                    })
                else:
                    print(f"Warning: Expression '{expression}' from {page_title} not found in Stage B scores. Skipping this expression.")
                    
        except Exception as e:
            print(f"Error processing page {page_title}: {e}")
            continue
            
    if not all_training_samples:
        print("No training samples could be loaded. Ensure data exists and paths are correct.")
        return pd.DataFrame()

    return pd.DataFrame(all_training_samples)


def train_and_save_model(training_df: pd.DataFrame, model_output_path: str):
    """
    Trains a classification model and saves it.

    Args:
        training_df: DataFrame with 'similarity_score' (feature) and 'human_rank' (label).
        model_output_path: Path to save the trained model.
    """
    if training_df.empty:
        print("Training data is empty. Cannot train model.")
        return

    X = training_df[['similarity_score']] # Features
    y = training_df['human_rank']         # Labels

    if len(X) < 10: # Arbitrary small number, adjust as needed
        print(f"Warning: Very few training samples ({len(X)}). Model quality may be poor.")
    
    if y.nunique() < 2:
        print(f"Error: Need at least 2 unique classes for training, but found {y.nunique()}. Cannot train model.")
        return

    # Split data (optional, but good for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

    print(f"Training model with {len(X_train)} samples, testing with {len(X_test)} samples.")
    print(f"Class distribution in training data:\n{y_train.value_counts(normalize=True)}")
    print(f"Class distribution in test data:\n{y_test.value_counts(normalize=True)}")


    # Initialize and train the model (RandomForestClassifier as an example)
    # You might want to tune hyperparameters or try different models.
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during model training: {e}")
        print("This might be due to insufficient samples for some classes after stratification.")
        return

    # Evaluate the model (optional)
    if X_test.shape[0] > 0:
        y_pred = model.predict(X_test)
        print("\nModel Evaluation on Test Set:")
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print("No test data to evaluate.")

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nTrained ranker model saved to {model_output_path}")


def main():
    """
    Main function to load data, train the ranker model, and save it.
    """
    # Configuration
    ranked_pages_dir = "data/ranked_pages"  # Directory with human-ranked CSVs
    stage_b_output_dir = "data/processed/stage_b" # Directory with Stage B similarity scores
    model_output_path = "models/stage_c_ranker.joblib" # Path to save the trained model

    print("Starting Stage C model training...")
    
    # Load training data
    training_df = load_training_data(ranked_pages_dir, stage_b_output_dir)

    if training_df.empty:
        print("Failed to load training data. Exiting.")
        return

    print(f"\nLoaded {len(training_df)} training samples.")
    print(f"Sample data:\n{training_df.head()}")
    print(f"Rank distribution in loaded data:\n{training_df['human_rank'].value_counts().sort_index()}")


    # Train and save the model
    train_and_save_model(training_df, model_output_path)
    
    print("\nStage C model training finished.")

if __name__ == "__main__":
    main()
