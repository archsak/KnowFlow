import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
import joblib
from typing import List, Dict, Tuple
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download nltk resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load spacy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class FeatureExtractor:
    """
    Extracts features from documents and expressions for ranking model training
    """
    
    def __init__(self):
        """Initialize feature extractor with required NLP resources"""
        self.stop_words = set(stopwords.words('english'))
    
    def extract_features(self, 
                         document_text: str, 
                         expression: str,
                         similarity_score: float = None) -> Dict[str, float]:
        """
        Extract features for an expression from the document
        
        Args:
            document_text: Full text of the document
            expression: The expression to extract features for
            similarity_score: Optional similarity score from Stage B
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Clean and normalize text for accurate matching
        doc_text_lower = document_text.lower()
        expr_lower = expression.lower()
        
        # Basic expression properties
        features['expression_length'] = len(expression)
        features['expression_word_count'] = len(expression.split())
        
        # Document-expression relationship features
        features['appears_in_document'] = 1 if expr_lower in doc_text_lower else 0
        
        # Frequency features
        if features['appears_in_document']:
            # Simple count (may count substrings unintentionally)
            simple_count = doc_text_lower.count(expr_lower)
            
            # More accurate count with word boundaries
            # This regex looks for the expression with word boundaries
            pattern = r'\b' + re.escape(expr_lower) + r'\b'
            accurate_count = len(re.findall(pattern, doc_text_lower))
            
            features['occurrence_count'] = accurate_count
            features['occurrence_density'] = accurate_count / (len(document_text.split()) + 0.001)
            
            # Position features
            first_pos = doc_text_lower.find(expr_lower)
            if first_pos >= 0:
                features['first_occurrence_position'] = first_pos / len(document_text)
                
                # Check if expression appears in first paragraph
                first_paragraph_end = document_text.find('\n\n')
                if first_paragraph_end == -1:  # If no double newline, use first 500 chars as approximation
                    first_paragraph_end = min(500, len(document_text))
                features['in_first_paragraph'] = 1 if first_pos < first_paragraph_end else 0
                
                # Check if expression appears in first N% of document
                features['in_first_10_percent'] = 1 if first_pos < (len(document_text) * 0.1) else 0
                features['in_first_25_percent'] = 1 if first_pos < (len(document_text) * 0.25) else 0
                
                # Find all occurrences
                # This regex counts non-overlapping occurrences with word boundaries
                all_matches = [m.start() for m in re.finditer(pattern, doc_text_lower)]
                
                if all_matches:
                    # Average position of all occurrences
                    avg_pos = sum(all_matches) / (len(all_matches) * len(document_text))
                    features['avg_occurrence_position'] = avg_pos
                    
                    # Distribution of occurrences
                    if len(all_matches) > 1:
                        # Standard deviation of positions, normalized by document length
                        std_pos = np.std([pos / len(document_text) for pos in all_matches])
                        features['position_std'] = std_pos
                        
                        # Calculate if occurrences are clustered or spread throughout
                        # Position spread = (last position - first position) / document length
                        position_spread = (all_matches[-1] - all_matches[0]) / len(document_text)
                        features['position_spread'] = position_spread
                    else:
                        features['position_std'] = 0
                        features['position_spread'] = 0
            else:
                features['first_occurrence_position'] = -1
                features['in_first_paragraph'] = 0
                features['in_first_10_percent'] = 0
                features['in_first_25_percent'] = 0
                features['avg_occurrence_position'] = -1
                features['position_std'] = -1
                features['position_spread'] = -1
        else:
            features['first_occurrence_position'] = -1
            features['in_first_paragraph'] = 0
            features['in_first_10_percent'] = 0
            features['in_first_25_percent'] = 0
            features['avg_occurrence_position'] = -1
            features['position_std'] = -1
            features['position_spread'] = -1
        
        # Linguistic features using spaCy
        doc = nlp(expression)
        
        # Part of speech features
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
        # Some common POS tags
        for pos in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
            features[f'pos_{pos}_count'] = pos_counts.get(pos, 0)
            features[f'pos_{pos}_ratio'] = pos_counts.get(pos, 0) / max(len(doc), 1)
            
        # Named entity recognition
        is_entity = any(ent.text for ent in doc.ents)
        features['is_named_entity'] = 1 if is_entity else 0
        
        # If similarity score is provided from Stage B
        if similarity_score is not None:
            features['similarity_score'] = similarity_score
        
        return features

# def load_training_data(ranked_pages_dir: str, 
#                        raw_data_dir: str,
#                        stage_b_output_dir: str = None) -> pd.DataFrame:
#     """
#     Loads human-ranked expressions, raw text, and optionally similarity scores
#     from Stage B to create a training dataset with rich features.

#     Args:
#         ranked_pages_dir: Directory containing CSV files with human-ranked expressions
#                           (columns: 'expression', 'rank'). File names are page titles.
#         raw_data_dir: Directory containing raw text files for documents.
#         stage_b_output_dir: Optional directory with Stage B similarity scores.

#     Returns:
#         A pandas DataFrame with features and human ranks for training.
#     """
#     all_training_samples = []
#     feature_extractor = FeatureExtractor()

#     if not os.path.exists(ranked_pages_dir):
#         print(f"Error: Ranked pages directory not found: {ranked_pages_dir}")
#         return pd.DataFrame()
    
#     if not os.path.exists(raw_data_dir):
#         print(f"Error: Raw data directory not found: {raw_data_dir}")
#         return pd.DataFrame()

#     for ranked_file_name in os.listdir(ranked_pages_dir):
#         if not ranked_file_name.endswith(".csv"):
#             continue
        
#         page_title = ranked_file_name.replace(".csv", "")
#         ranked_file_path = os.path.join(ranked_pages_dir, 'rated_wiki_pages.csv')
#         #ranked_file_path = os.path.join(ranked_pages_dir, ranked_file_name)
#         raw_file_path = os.path.join(raw_data_dir, f"{page_title}.txt")
        
#         # Skip if raw document text is not available
#         if not os.path.exists(raw_file_path):
#             print(f"Warning: Raw text file for {page_title} not found at {raw_file_path}. Skipping.")
#             continue
            
#         # Load raw document text
#         try:
#             with open(raw_file_path, 'r', encoding='utf-8') as f:
#                 document_text = f.read()
#         except Exception as e:
#             print(f"Error reading document text for {page_title}: {e}")
#             continue
            
#         # Load similarity scores if available
#         similarity_scores = None
#         if stage_b_output_dir:
#             stage_b_file_path = os.path.join(stage_b_output_dir, f"{page_title}_filtered.json")
#             if os.path.exists(stage_b_file_path):
#                 try:
#                     with open(stage_b_file_path, 'r', encoding='utf-8') as f:
#                         similarity_scores = json.load(f)
#                 except Exception as e:
#                     print(f"Error loading Stage B scores for {page_title}: {e}")

#         # Load human-ranked data
#         try:
#             human_ranks_df = pd.read_csv(ranked_file_path)
#             if not ({'concept', 'score'}.issubset(human_ranks_df.columns)):
#                 print(f"Warning: CSV file {ranked_file_name} is missing required columns. Skipping.")
#                 continue
                
#             # Process each expression
#             for _, row in human_ranks_df.iterrows():
#                 expression = row['concept']
#                 human_rank = int(row['score'])  # Ensure rank is integer
                
#                 # Get similarity score if available
#                 similarity_score = None
#                 if similarity_scores and expression in similarity_scores:
#                     similarity_score = similarity_scores[expression]
                
#                 # Extract features
#                 features = feature_extractor.extract_features(
#                     document_text=document_text,
#                     expression=expression,
#                     similarity_score=similarity_score
#                 )
                
#                 # Add human rank and expression info to features
#                 features['expression'] = expression
#                 features['page_title'] = page_title
#                 features['human_rank'] = human_rank
                
#                 all_training_samples.append(features)
                
#         except Exception as e:
#             print(f"Error processing human ranks for {page_title}: {e}")
#             continue
            
#     if not all_training_samples:
#         print("No training samples could be loaded. Ensure data exists and paths are correct.")
#         return pd.DataFrame()

#     # Convert to DataFrame
#     training_df = pd.DataFrame(all_training_samples)
    
#     # Handle missing values
#     training_df = training_df.fillna(-1)
#     training_df.to_csv("data/processed/training_data.csv", index=False)

    
#     return training_df

def load_training_data(ranked_csv_path: str, 
                       raw_data_dir: str,
                       stage_b_output_dir: str = None) -> pd.DataFrame:
    """
    Loads human-ranked expressions from a single CSV file, raw text documents,
    and optional similarity scores to create training data.

    Args:
        ranked_csv_path: Path to the combined CSV file of human-ranked expressions.
        raw_data_dir: Directory containing raw text files (one per article).
        stage_b_output_dir: Optional directory containing Stage B similarity scores.

    Returns:
        A pandas DataFrame with extracted features and labels.
    """
    all_training_samples = []
    feature_extractor = FeatureExtractor()

    if not os.path.exists(ranked_csv_path):
        print(f"Error: Ranked CSV not found at {ranked_csv_path}")
        return pd.DataFrame()

    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw text folder not found: {raw_data_dir}")
        return pd.DataFrame()

    try:
        ranked_df = pd.read_csv(ranked_csv_path)
    except Exception as e:
        print(f"Failed to load ranked CSV: {e}")
        return pd.DataFrame()

    # Validate expected columns
    if not {'source_article', 'concept', 'score'}.issubset(ranked_df.columns):
        print("Error: CSV must contain 'source_article', 'concept', and 'score' columns.")
        return pd.DataFrame()

    for page_title in ranked_df['source_article'].unique():
        raw_file_name = f"{page_title.replace(' ', '_')}.txt"
        raw_file_path = os.path.join(raw_data_dir, raw_file_name)

        if not os.path.exists(raw_file_path):
            print(f"Warning: Raw text for {page_title} not found at {raw_file_path}. Skipping.")
            continue

        try:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
        except Exception as e:
            print(f"Error reading raw text for {page_title}: {e}")
            continue

        # Load similarity scores if available
        similarity_scores = None
        if stage_b_output_dir:
            stage_b_file_path = os.path.join(stage_b_output_dir, f"{page_title}_filtered.json")
            if os.path.exists(stage_b_file_path):
                try:
                    with open(stage_b_file_path, 'r', encoding='utf-8') as f:
                        similarity_scores = json.load(f)
                except Exception as e:
                    print(f"Error loading Stage B scores for {page_title}: {e}")

        article_rows = ranked_df[ranked_df['source_article'] == page_title]
        for _, row in article_rows.iterrows():
            expression = row['concept']
            human_rank = int(row['score'])

            similarity_score = None
            if similarity_scores and expression in similarity_scores:
                similarity_score = similarity_scores[expression]

            features = feature_extractor.extract_features(
                document_text=document_text,
                expression=expression,
                similarity_score=similarity_score
            )

            features['expression'] = expression
            features['page_title'] = page_title
            features['human_rank'] = human_rank

            all_training_samples.append(features)

    if not all_training_samples:
        print("No training samples loaded.")
        return pd.DataFrame()

    loaded_df = pd.DataFrame(all_training_samples)
    loaded_df = loaded_df.fillna(-1)
    return loaded_df


# def train_and_save_model(loaded_df: pd.DataFrame, model_output_path: str, perform_grid_search: bool = False):
#     """
#     Trains a classification model and saves it.

#     Args:
#         loaded_df: DataFrame with features and 'human_rank' as the label.
#         model_output_path: Path to save the trained model.
#         perform_grid_search: Whether to perform hyperparameter tuning with GridSearchCV.
#     """
#     if loaded_df.empty:
#         print("Training data is empty. Cannot train model.")
#         return

#     # Separate features and target
#     feature_columns = [col for col in loaded_df.columns 
#                       if col not in ['expression', 'page_title', 'human_rank']]
    
#     print(f"\nUsing {len(feature_columns)} features for training: {feature_columns}")
    
#     X = loaded_df[feature_columns]
#     y = loaded_df['human_rank']

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     print(f"\nTraining with {len(X_train)} samples, testing with {len(X_test)} samples.")
#     print(f"Class distribution in training data:\n{y_train.value_counts(normalize=True)}")

#     # Train model
#     if perform_grid_search:
#         # Grid search for hyperparameter tuning
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [None, 10, 20, 30],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4]
#         }
        
#         print("\nPerforming grid search for hyperparameter tuning...")
#         grid_search = GridSearchCV(
#             RandomForestClassifier(random_state=42, class_weight='balanced'),
#             param_grid=param_grid,
#             cv=5,
#             scoring='f1_weighted',
#             n_jobs=-1
#         )
        
#         grid_search.fit(X_train, y_train)
#         model = grid_search.best_estimator_
#         print(f"Best parameters: {grid_search.best_params_}")
        
#     else:
#         # Use default RandomForestClassifier with some reasonable settings
#         model = RandomForestClassifier(
#             n_estimators=100,
#             random_state=42,
#             class_weight='balanced'
#         )
#         model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
    
#     print("\nModel Evaluation on Test Set:")
#     print(classification_report(y_test, y_pred, zero_division=0))
    
#     # Feature importance
#     feature_importance = pd.DataFrame({
#         'Feature': feature_columns,
#         'Importance': model.feature_importances_
#     }).sort_values(by='Importance', ascending=False)
    
#     print("\nTop 10 Most Important Features:")
#     print(feature_importance.head(10))
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print("\nConfusion Matrix:")
#     print(cm)
#     # Save confusion matrix as CSV
#     os.makedirs('results', exist_ok=True)
#     np.savetxt('results/confusion_matrix.csv', cm, delimiter=",", fmt='%d')
#     print("\nConfusion matrix saved to 'results/confusion_matrix.csv'")
    
#     # Save feature importances
#     feature_importance.to_csv('results/feature_importance.csv', index=False)
#     print("Feature importance saved to 'results/feature_importance.csv'")
    
#     # Save the trained model
#     os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
#     joblib.dump(model, model_output_path)
    
#     # Save the feature columns for future use
#     feature_columns_path = os.path.join(os.path.dirname(model_output_path), 'feature_columns.json')
#     with open(feature_columns_path, 'w') as f:
#         json.dump(feature_columns, f)
        
#     print(f"\nTrained ranker model saved to {model_output_path}")
#     print(f"Feature columns list saved to {feature_columns_path}")


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split

def train_and_save_model(loaded_df: pd.DataFrame, model_output_path: str, perform_grid_search: bool = False):
    """
    Trains a regression model (RandomForestRegressor), evaluates using MSE, and saves it.

    Args:
        loaded_df: DataFrame with features and 'human_rank' as the target.
        model_output_path: Path to save the trained model.
        perform_grid_search: Unused for now.
    """
    if loaded_df.empty:
        print("Training data is empty. Cannot train model.")
        return

    # Separate features and target
    feature_columns = [col for col in loaded_df.columns 
                      if col not in ['expression', 'page_title', 'human_rank']]
    
    
    X = loaded_df[feature_columns]
    y = loaded_df['human_rank']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Train regressor
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict and round to nearest integer
    y_pred_continuous = model.predict(X_test)
    y_pred_rounded = np.round(y_pred_continuous).astype(int)

    # Evaluate with MSE
    mse = mean_squared_error(y_test, y_pred_rounded)
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    
    accuracy = accuracy_score(y_test, y_pred_rounded)
    print(f"Accuracy (rounded predictions match true class): {accuracy:.4f}")



    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)

    print(f"\nTrained ranker model saved to {model_output_path}")



def main():
    """
    Main function to load data, train the ranker model, and save it.
    """
    # Configuration
    ranked_csv_path = "data/raw/ranked_pages/rated_wiki_pages.csv"  # Directory with human-ranked CSVs
    raw_data_dir = "data/raw/raw_texts"  # Directory with raw text files
    stage_b_output_dir = "data/processed/stage_b"  # Optional: Directory with Stage B similarity scores
    model_output_path = "models/stage_c_ranker.joblib"  # Path to save the trained model
    perform_grid_search = False  # Set to True to perform hyperparameter tuning (takes longer)

    print("Starting Stage C model training...")
    print(f"Loading data from ranked pages directory: {ranked_csv_path}")
    print(f"Loading raw text from directory: {raw_data_dir}")
    print(f"Using similarity scores from Stage B (optional): {stage_b_output_dir}")
    
    # Create required directories
    os.makedirs("models", exist_ok=True)
    
    # Load and prepare training data with features
    training_df = load_training_data(
        ranked_csv_path=ranked_csv_path,
        raw_data_dir=raw_data_dir,
        stage_b_output_dir=stage_b_output_dir  # Can be None if you don't want to use similarity scores
    )

    if training_df.empty:
        print("Failed to load training data. Exiting.")
        return

    # print(f"\nLoaded {len(training_df)} training samples.")
    # print(f"Rank distribution in loaded data:\n{training_df['human_rank'].value_counts().sort_index()}")
    
    # # Show a few examples
    # print("\nSample data (first few rows with selected columns):")
    # sample_columns = ['expression', 'expression_length', 'occurrence_count', 'human_rank']
    # available_columns = [col for col in sample_columns if col in training_df.columns]
    # print(training_df[available_columns].head())

    # Train and save the model
    train_and_save_model(training_df, model_output_path, perform_grid_search)
    
    print("\nStage C model training finished.")

if __name__ == "__main__":
    main()
