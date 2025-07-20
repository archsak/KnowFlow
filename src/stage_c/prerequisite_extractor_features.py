import os
import json
from typing import List, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
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
    Extracts features from documents and expressions for ranking model.
    Must match the feature extraction used in training.
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
                pattern = r'\b' + re.escape(expr_lower) + r'\b'
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
        return features

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
            
        model_dir = os.path.dirname(model_path)
        feature_columns_path = os.path.join(model_dir, 'feature_columns.json')
        
        try:
            # Load the model
            self.model = joblib.load(model_path)
            print(f"Prerequisite ranking model loaded from {model_path}")
            
            # Load feature columns used during training
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
                print(f"Loaded {len(self.feature_columns)} feature columns")
            else:
                # If feature columns file doesn't exist, we'll have to infer from the model
                print(f"Warning: Feature columns file not found at {feature_columns_path}")
                print("Will attempt to infer feature requirements from the model")
                self.feature_columns = None
                
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor()
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")

    def rank_expressions(self, 
                         filtered_expressions: Dict[str, float], 
                         document_text: str) -> Dict[str, int]:
        """
        Rank filtered expressions by extracting features and predicting scores.

        Args:
            filtered_expressions: Dictionary mapping expressions to their similarity scores (from Stage B).
            document_text: The full text of the document.

        Returns:
            Dictionary mapping expressions to their predicted importance scores (0-3).
        """
        if not filtered_expressions:
            return {}
            
        ranked_expressions = {}
        all_features = []
        expressions_list = list(filtered_expressions.keys())
        
        # Extract features for each expression
        for expr in expressions_list:
            features = self.feature_extractor.extract_features(
                document_text=document_text,
                expression=expr
            )
            all_features.append(features)
            
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame(all_features)
        
        # Ensure we have all required columns in the right order
        if self.feature_columns:
            # Add missing columns with default value -1
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = -1
                    
            # Select only the columns used during training, in the right order
            features_df = features_df[self.feature_columns]
        
        # Handle any missing values
        features_df = features_df.fillna(-1)
        
        try:
            # Predict ranks
            predicted_ranks = self.model.predict(features_df)
            
            # Assign ranks to expressions
            for i, expr in enumerate(expressions_list):
                rank = int(predicted_ranks[i])
                # Ensure rank is in valid range
                rank = max(0, min(3, rank)) 
                ranked_expressions[expr] = rank
                
            # Sort by predicted importance score (descending),
            # then by original similarity (descending) as a tie-breaker
            sorted_ranked_expressions = dict(sorted(
                ranked_expressions.items(),
                key=lambda item: (item[1], filtered_expressions[item[0]]),
                reverse=True
            ))
            return sorted_ranked_expressions
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback: return all expressions with default rank 0
            return {expr: 0 for expr in expressions_list}

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
                         ranker: PrerequisiteRanker) -> Dict[str, int]:
    """
    Process a single page: load raw text and Stage B results, rank expressions, save Stage C results.
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

    # Rank expressions using document text and similarity scores
    ranked_expressions = ranker.rank_expressions(filtered_expressions, document_text)

    # Save results
    output_path = os.path.join(stage_c_output_dir, f"{page_title}_ranked_prerequisites.json")
    save_ranked_expressions(ranked_expressions, output_path)

    return ranked_expressions

def main():
    """
    Main function to process all pages for prerequisite ranking.
    """
    # Configuration
    raw_data_dir = "data/raw"
    stage_b_output_dir = "data/processed/stage_b" 
    stage_c_output_dir = "data/processed/stage_c"
    
    # Path to the trained ranker model
    model_path = "models/stage_c_ranker.joblib" 

    if not os.path.exists(model_path):
        print(f"Error: Ranker model not found at {model_path}.")
        print("Please ensure `src/stage_c/train_ranker.py` has been run and the model is saved correctly.")
        return

    # Initialize the prerequisite ranker
    try:
        ranker = PrerequisiteRanker(model_path=model_path)
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
