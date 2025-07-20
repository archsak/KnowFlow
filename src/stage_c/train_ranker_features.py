import os
import sys
import re
import json
from typing import Dict
import joblib
import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Add src to path to import get_raw_text
sys.path.append('src')
from stage_a.Bert1 import get_raw_text

# --- NLP Resource Loading ---
def download_nlp_resources():
    """Download necessary NLTK and spaCy resources if not found."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        os.system("python -m spacy download en_core_web_sm")

download_nlp_resources()
nlp = spacy.load("en_core_web_sm")

# --- Feature Extraction ---
class FeatureExtractor:
    """Extracts rich features for prerequisite ranking."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_features(self, 
                         document_text: str, 
                         expression: str,
                         similarity_score: float = None) -> Dict[str, float]:
        """Extracts a dictionary of features for a given expression and document."""
        features = {}
        doc_text_lower = document_text.lower()
        expr_lower = expression.lower()
        
        # Basic features
        features['expression_length'] = len(expression)
        features['expression_word_count'] = len(expression.split())
        
        # Occurrence and position features
        pattern = r'\b' + re.escape(expr_lower) + r'\b'
        all_matches = [m.start() for m in re.finditer(pattern, doc_text_lower)]
        
        features['occurrence_count'] = len(all_matches)
        
        if features['occurrence_count'] > 0:
            doc_word_count = len(document_text.split())
            features['occurrence_density'] = features['occurrence_count'] / (doc_word_count + 1e-6)
            
            first_pos = all_matches[0]
            features['first_occurrence_position'] = first_pos / len(document_text)
            
            # Positional indicators
            features['in_first_10_percent'] = 1 if features['first_occurrence_position'] < 0.1 else 0
            features['in_first_25_percent'] = 1 if features['first_occurrence_position'] < 0.25 else 0
            
            if len(all_matches) > 1:
                features['position_spread'] = (all_matches[-1] - first_pos) / len(document_text)
                normalized_positions = [pos / len(document_text) for pos in all_matches]
                features['position_std'] = np.std(normalized_positions)
            else:
                features['position_spread'] = 0
                features['position_std'] = 0
        else:
            # Default values if expression is not found
            features['occurrence_density'] = 0
            features['first_occurrence_position'] = -1
            features['in_first_10_percent'] = 0
            features['in_first_25_percent'] = 0
            features['position_spread'] = -1
            features['position_std'] = -1

        # Linguistic features using spaCy
        doc = nlp(expression)
        pos_counts = doc.count_by(spacy.attrs.POS)
        
        for pos_id, count in pos_counts.items():
            pos_name = doc.vocab.strings[pos_id]
            features[f'pos_{pos_name}_count'] = count
        
        features['is_named_entity'] = 1 if doc.ents else 0
        
        # Optional similarity score from Stage B
        return features

# --- Data Loading and Splitting ---
def load_training_data(ranked_csv_path: str,
                       raw_data_dir: str, # Note: raw_data_dir is unused, but kept for signature consistency
                       stage_b_output_dir: str = None) -> pd.DataFrame:
    """
    Loads human-ranked expressions from a single CSV file and fetches raw text
    to create training data.
    """
    all_training_samples = []

    if not os.path.exists(ranked_csv_path):
        print(f"Error: Ranked CSV not found at {ranked_csv_path}")
        return pd.DataFrame()

    ranked_df = pd.read_csv(ranked_csv_path)

    if not {'source_article', 'concept', 'score'}.issubset(ranked_df.columns):
        print("Error: CSV must contain 'source_article', 'concept', and 'score' columns.")
        return pd.DataFrame()

    # --- Efficient Text Caching ---
    print("Caching article texts to prevent redundant fetching...")
    unique_articles = ranked_df['source_article'].unique()
    article_texts = {}
    for title in unique_articles:
        try:
            article_texts[title] = get_raw_text(title)
        except Exception as e:
            print(f"Warning: Could not fetch text for '{title}'. It will be skipped. Error: {e}")
    
    print(f"Cached {len(article_texts)} article texts.")

    for _, row in ranked_df.iterrows():
        page_title = row['source_article']
        
        if page_title not in article_texts:
            continue
            
        document_text = article_texts[page_title]

        all_training_samples.append({
            'expression': row['concept'],
            'page_title': page_title,
            'document_text': document_text,
            'human_rank': int(row['score']),
        })

    if not all_training_samples:
        print("No training samples could be loaded.")
        return pd.DataFrame()

    return pd.DataFrame(all_training_samples)


def load_and_prepare_data(ranked_csv_path: str) -> pd.DataFrame:
    """Loads and prepares the feature-rich dataset."""
    all_samples = []
    feature_extractor = FeatureExtractor()

    if not os.path.exists(ranked_csv_path):
        print(f"Error: Ranked CSV not found at {ranked_csv_path}")
        return pd.DataFrame()

    ranked_df = pd.read_csv(ranked_csv_path)
    
    # --- Efficient Text Caching ---
    print("Caching article texts to prevent redundant fetching...")
    unique_articles = ranked_df['source_article'].unique()
    article_texts = {}
    for title in unique_articles:
        try:
            article_texts[title] = get_raw_text(title)
        except Exception as e:
            print(f"Warning: Could not fetch text for '{title}'. It will be skipped. Error: {e}")
    
    print(f"Cached {len(article_texts)} article texts.")

    print("\nExtracting features for all samples...")
    for _, row in ranked_df.iterrows():
        page_title = row['source_article']
        expression = row['concept']
        
        # Skip if text could not be fetched
        if page_title not in article_texts:
            continue
            
        document_text = article_texts[page_title]
        
        features = feature_extractor.extract_features(document_text, expression)
        features['expression'] = expression
        features['page_title'] = page_title
        features['human_rank'] = int(row['score'])
        all_samples.append(features)

    return pd.DataFrame(all_samples).fillna(0)

def split_data_by_articles(df, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """Splits data by articles to prevent data leakage."""
    unique_articles = df['page_title'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_articles)
    
    n_articles = len(unique_articles)
    train_end = int(n_articles * train_ratio)
    val_end = int(n_articles * (train_ratio + val_ratio))
    
    train_articles = unique_articles[:train_end]
    val_articles = unique_articles[train_end:val_end]
    test_articles = unique_articles[val_end:]
    
    train_df = df[df['page_title'].isin(train_articles)]
    val_df = df[df['page_title'].isin(val_articles)]
    test_df = df[df['page_title'].isin(test_articles)]
    
    print("\nData split by articles:")
    print(f"Train: {len(train_articles)} articles, {len(train_df)} samples")
    print(f"Validation: {len(val_articles)} articles, {len(val_df)} samples")
    print(f"Test: {len(test_articles)} articles, {len(test_df)} samples")
    
    return train_df, val_df, test_df

# --- Model Evaluation ---
def evaluate_model(model, df, feature_extractor):
    """Evaluates the feature-based model."""
    if df.empty:
        return pd.DataFrame()

    all_features = []
    # Ensure 'document_text' column exists
    if 'document_text' not in df.columns:
        raise ValueError("Input DataFrame for evaluation must contain a 'document_text' column.")

    for _, row in df.iterrows():
        features = feature_extractor.extract_features(
            document_text=row['document_text'],
            expression=row['expression']
        )
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features).fillna(0)
    
    # Align columns with model's expected features
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[model_features]

    predictions = model.predict(features_df)
    
    # Round and clip predictions
    rounded_predictions = np.clip(np.round(predictions).astype(int), 0, 3)
    
    result_df = df.copy()
    result_df['predicted_rank'] = rounded_predictions
    
    return result_df


def detailed_prediction_analysis(predictions: np.ndarray, labels: np.ndarray, model_name: str):
    """Provides a detailed analysis of prediction results and rounding methods."""
    print(f"\n--- {model_name}: Detailed Test Set Analysis ---")
    
    # Continuous prediction stats
    print(f"Continuous Prediction Stats: Min={predictions.min():.3f}, Max={predictions.max():.3f}, Mean={predictions.mean():.3f}")

    # Analyze different rounding methods
    methods = {'round': np.round, 'floor': np.floor, 'ceil': np.ceil}
    best_accuracy = 0
    best_method = ''
    
    print("\nRounding Method Comparison:")
    for name, method in methods.items():
        rounded_preds = np.clip(method(predictions).astype(int), 0, 3)
        accuracy = accuracy_score(labels, rounded_preds)
        mse = mean_squared_error(labels, rounded_preds)
        print(f"Method: {name:5s} | Accuracy: {accuracy:.4f} | MSE: {mse:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = name
            
    print(f"\nBest rounding method: '{best_method}' with accuracy {best_accuracy:.4f}")
    
    # Final evaluation with the best method
    final_preds = np.clip(methods[best_method](predictions).astype(int), 0, 3)
    
    print("\nDistribution of Final Predictions vs. Labels:")
    pred_counts = pd.Series(final_preds).value_counts().sort_index()
    label_counts = pd.Series(labels).value_counts().sort_index()
    dist_df = pd.DataFrame({'predictions': pred_counts, 'labels': label_counts}).fillna(0).astype(int)
    print(dist_df)
    
    return best_method

# --- Main Training Logic ---
def train_and_save_model(df: pd.DataFrame, model_output_path: str):
    """Trains, evaluates, and saves the feature-based model."""
    if df.empty:
        print("Training data is empty. Aborting.")
        return

    train_df, val_df, test_df = split_data_by_articles(df)
    
    feature_columns = [col for col in df.columns if col not in ['expression', 'page_title', 'human_rank']]
    print(f"\nTraining with {len(feature_columns)} features.")
    
    X_train = train_df[feature_columns]
    y_train = train_df['human_rank']
    X_val = val_df[feature_columns]
    y_val = val_df['human_rank']
    X_test = test_df[feature_columns]
    y_test = test_df['human_rank']

    # Train model
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    print("Training RandomForestRegressor model...")
    model.fit(X_train, y_train)

    # --- Evaluation ---
    # Validation set
    val_preds_continuous = model.predict(X_val)
    val_preds_rounded = np.clip(np.round(val_preds_continuous).astype(int), 0, 3)
    val_accuracy = accuracy_score(y_val, val_preds_rounded)
    print(f"\nValidation Accuracy (standard rounding): {val_accuracy:.4f}")

    # Test set
    test_preds_continuous = model.predict(X_test)
    detailed_prediction_analysis(test_preds_continuous, y_test.values, "Feature-based RF Model")

    # --- Save Model ---
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to {model_output_path}")
    
    # Save feature columns for consistency
    feature_columns_path = os.path.join(os.path.dirname(model_output_path), 'features_columns.json')
    with open(feature_columns_path, 'w') as f:
        json.dump(feature_columns, f)
    print(f"Feature columns saved to {feature_columns_path}")

def main():
    """Main execution function."""
    print("=== Stage C: Feature-based Prerequisite Ranker Training ===")
    
    # Configuration
    ranked_csv_path = "data/raw/ranked_pages/rated_wiki_pages.csv"
    model_output_path = "models/stage_c_ranker_features.joblib"

    # Run pipeline
    master_df = load_and_prepare_data(ranked_csv_path)
    train_and_save_model(master_df, model_output_path)
    
    print("\n--- Training complete ---")

if __name__ == "__main__":
    main()
