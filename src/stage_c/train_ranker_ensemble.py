import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import re
import nltk
import spacy

# Import custom modules
from stage_c.train_ranker_features import FeatureExtractor, load_training_data
from stage_c.prerequisite_extractor_features import PrerequisiteRanker
from prerequisite_extractor_encoder import PrerequisiteRankerEncoder

class EnsemblePrerequisiteRanker:
    """
    An ensemble ranker that combines predictions from both feature-based and encoder-based
    models to produce improved ranking of prerequisite expressions.
    """
    
    def __init__(
        self, 
        feature_model_path: str = None, 
        encoder_model_path: str = None,
        meta_learner: object = None,
        device: str = None
    ):
        """
        Initialize the ensemble ranker with paths to both models.
        
        Args:
            feature_model_path: Path to the feature-based model
            encoder_model_path: Path to the encoder-based model
            meta_learner: Optional pre-trained meta-learner model
            device: Device to run the encoder model on ('cuda' or 'cpu')
        """
        self.feature_model = None
        self.encoder_model = None
        self.meta_learner = meta_learner
        self.feature_extractor = FeatureExtractor()
        
        # Set device for encoder model
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load feature-based model if provided
        if feature_model_path:
            try:
                self.feature_model = joblib.load(feature_model_path)
                
                # Load feature columns
                model_dir = os.path.dirname(feature_model_path)
                feature_columns_path = os.path.join(model_dir, 'feature_columns.json')
                if os.path.exists(feature_columns_path):
                    with open(feature_columns_path, 'r') as f:
                        self.feature_columns = json.load(f)
                else:
                    self.feature_columns = None
                    print("Warning: Feature columns file not found")
                
                print(f"Feature-based model loaded from {feature_model_path}")
            except Exception as e:
                print(f"Error loading feature-based model: {e}")
        
        # Load encoder-based model if provided
        if encoder_model_path:
            try:
                self.encoder_model = PrerequisiteRankerEncoder(
                    model_path=encoder_model_path, 
                    device=self.device
                )
                print(f"Encoder-based model loaded from {encoder_model_path}")
            except Exception as e:
                print(f"Error loading encoder-based model: {e}")
                
        # Initialize meta-learner if not provided
        if meta_learner is None and (self.feature_model or self.encoder_model):
            self.meta_learner = LogisticRegression(
                multi_class='multinomial', 
                solver='lbfgs', 
                max_iter=1000,
                class_weight='balanced'
            )
    
    def get_feature_model_probas(self, document_text: str, expressions: List[str], 
                             similarity_scores: Dict[str, float] = None) -> np.ndarray:
        """
        Get probability predictions from the feature-based model.
        
        Args:
            document_text: Text of the document
            expressions: List of expressions to rank
            similarity_scores: Optional dictionary of similarity scores
            
        Returns:
            Array of probability predictions for each expression
        """
        if self.feature_model is None:
            return None
        
        # Extract features for each expression
        features_list = []
        for expr in expressions:
            sim_score = similarity_scores.get(expr) if similarity_scores else None
            features = self.feature_extractor.extract_features(
                document_text=document_text,
                expression=expr,
                similarity_score=sim_score
            )
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Ensure we have all required columns in the right order
        if self.feature_columns:
            # Add missing columns with default value -1
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = -1
                    
            # Select only the columns used during training, in the right order
            features_df = features_df[self.feature_columns]
        
        # Handle missing values
        features_df = features_df.fillna(-1)
        
        # Get probability predictions
        return self.feature_model.predict_proba(features_df)
    
    def get_encoder_model_probas(self, document_text: str, expressions: List[str]) -> np.ndarray:
        """
        Get probability predictions from the encoder-based model.
        
        Args:
            document_text: Text of the document
            expressions: List of expressions to rank
            
        Returns:
            Array of probability predictions for each expression
        """
        if self.encoder_model is None:
            return None
        
        # Create a dummy dictionary with expressions as keys (values don't matter for probas)
        dummy_scores = {expr: 1.0 for expr in expressions}
        
        # Use the encoder model's predict_proba method
        proba_dict = self.encoder_model.predict_proba(dummy_scores, document_text)
        
        # Convert dictionary to array, ensuring same order as input expressions
        probas = np.array([proba_dict[expr] for expr in expressions])
        return probas
    
    def train(self, document_texts: List[str], expressions: List[str], 
              ranks: List[int], similarity_scores: Optional[Dict[str, float]] = None):
        """
        Train the ensemble model by learning weights for each base model.
        
        Args:
            document_texts: List of document texts
            expressions: List of expressions to rank
            ranks: Ground truth ranks for each expression
            similarity_scores: Optional dictionary of similarity scores
        """
        if self.feature_model is None and self.encoder_model is None:
            print("Error: At least one base model (feature or encoder) must be loaded")
            return
            
        print("Getting predictions from base models...")
        meta_features = []
        
        # Process examples in batches
        batch_size = 32
        for i in range(0, len(expressions), batch_size):
            batch_end = min(i + batch_size, len(expressions))
            batch_expressions = expressions[i:batch_end]
            batch_docs = document_texts[i:batch_end]
            
            batch_features = []
            
            # Get feature model predictions
            if self.feature_model is not None:
                batch_sim_scores = {expr: similarity_scores.get(expr, 0.0) 
                                  for expr in batch_expressions} if similarity_scores else None
                
                for j, expr in enumerate(batch_expressions):
                    feature_probas = self.get_feature_model_probas(
                        batch_docs[j], [expr], 
                        {expr: batch_sim_scores.get(expr)} if batch_sim_scores else None
                    )
                    
                    if feature_probas is not None:
                        batch_features.append(feature_probas[0])
                    
            # Get encoder model predictions
            if self.encoder_model is not None:
                for j, expr in enumerate(batch_expressions):
                    encoder_probas = self.get_encoder_model_probas(batch_docs[j], [expr])
                    
                    if encoder_probas is not None:
                        # If we already have feature model probas, append to them
                        if self.feature_model is not None:
                            batch_features[j] = np.concatenate([batch_features[j], encoder_probas[0]])
                        else:
                            batch_features.append(encoder_probas[0])
            
            meta_features.extend(batch_features)
        
        # Convert to numpy array
        meta_features = np.array(meta_features)
        
        print(f"Training meta-learner on {len(meta_features)} examples with {meta_features.shape[1]} features")
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, ranks)
        
        # Print model coefficients to show relative importance
        if hasattr(self.meta_learner, 'coef_'):
            feature_model_classes = 4 if self.feature_model is not None else 0
            encoder_model_classes = 4 if self.encoder_model is not None else 0
            
            print("\nMeta-learner coefficients (weights for each model's predictions):")
            coefs = self.meta_learner.coef_
            
            if self.feature_model is not None and self.encoder_model is not None:
                feature_weight = np.mean(np.abs(coefs[:, :feature_model_classes]))
                encoder_weight = np.mean(np.abs(coefs[:, feature_model_classes:]))
                
                print(f"Average feature model weight: {feature_weight:.4f}")
                print(f"Average encoder model weight: {encoder_weight:.4f}")
                print(f"Ratio (encoder/feature): {encoder_weight/feature_weight:.4f}")
    
    def predict(self, document_text: str, expressions: List[str], 
               similarity_scores: Optional[Dict[str, float]] = None) -> Dict[str, int]:
        """
        Make predictions using the ensemble model.
        
        Args:
            document_text: Text of the document
            expressions: List of expressions to rank
            similarity_scores: Optional dictionary of similarity scores
            
        Returns:
            Dictionary mapping expressions to their predicted ranks
        """
        if not expressions:
            return {}
            
        if self.meta_learner is None:
            print("Error: Meta-learner not trained")
            return {expr: 0 for expr in expressions}
        
        # Get predictions from both models and combine them
        meta_features = []
        
        # Get feature model predictions
        if self.feature_model is not None:
            feature_probas = self.get_feature_model_probas(document_text, expressions, similarity_scores)
            if feature_probas is not None:
                meta_features.append(feature_probas)
        
        # Get encoder model predictions
        if self.encoder_model is not None:
            encoder_probas = self.get_encoder_model_probas(document_text, expressions)
            if encoder_probas is not None:
                meta_features.append(encoder_probas)
        
        # Combine features
        if not meta_features:
            print("Error: No predictions from base models")
            return {expr: 0 for expr in expressions}
            
        if len(meta_features) > 1:
            # If we have both models' predictions, concatenate them
            meta_features = np.hstack(meta_features)
        else:
            # Otherwise just use the one we have
            meta_features = meta_features[0]
        
        # Make predictions with meta-learner
        predictions = self.meta_learner.predict(meta_features)
        
        # Map predictions to expressions
        result = {expressions[i]: int(predictions[i]) for i in range(len(expressions))}
        
        # Sort by predicted rank (descending)
        sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_result
    
    def save(self, output_dir: str):
        """
        Save the ensemble model.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save meta-learner
        meta_learner_path = os.path.join(output_dir, 'ensemble_meta_learner.pkl')
        with open(meta_learner_path, 'wb') as f:
            pickle.dump(self.meta_learner, f)
        
        # Save configuration
        config = {
            'feature_model_present': self.feature_model is not None,
            'encoder_model_present': self.encoder_model is not None
        }
        
        config_path = os.path.join(output_dir, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        print(f"Ensemble model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str, feature_model_path: str = None, encoder_model_path: str = None):
        """
        Load a saved ensemble model.
        
        Args:
            model_dir: Directory containing the saved ensemble model
            feature_model_path: Path to the feature-based model
            encoder_model_path: Path to the encoder-based model
            
        Returns:
            Loaded EnsemblePrerequisiteRanker instance
        """
        # Load configuration
        config_path = os.path.join(model_dir, 'ensemble_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Ensemble configuration not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load meta-learner
        meta_learner_path = os.path.join(model_dir, 'ensemble_meta_learner.pkl')
        if not os.path.exists(meta_learner_path):
            raise FileNotFoundError(f"Meta-learner not found at {meta_learner_path}")
            
        with open(meta_learner_path, 'rb') as f:
            meta_learner = pickle.load(f)
        
        # Verify that required base models are available
        if config['feature_model_present'] and feature_model_path is None:
            raise ValueError("Feature model is required but path not provided")
            
        if config['encoder_model_present'] and encoder_model_path is None:
            raise ValueError("Encoder model is required but path not provided")
        
        # Create and return ensemble
        return cls(
            feature_model_path=feature_model_path if config['feature_model_present'] else None,
            encoder_model_path=encoder_model_path if config['encoder_model_present'] else None,
            meta_learner=meta_learner
        )


def load_encoder_data(ranked_pages_dir: str, raw_data_dir: str) -> pd.DataFrame:
    """
    Load data for encoder model training.
    
    Args:
        ranked_pages_dir: Directory with human-ranked CSVs
        raw_data_dir: Directory with raw text files
        
    Returns:
        DataFrame with expressions, documents, and ranks
    """
    all_data = []
    
    if not os.path.exists(ranked_pages_dir):
        print(f"Error: Ranked pages directory not found: {ranked_pages_dir}")
        return pd.DataFrame()
    
    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        return pd.DataFrame()
        
    for ranked_file_name in os.listdir(ranked_pages_dir):
        if not ranked_file_name.endswith(".csv"):
            continue
        
        page_title = ranked_file_name.replace(".csv", "")
        ranked_file_path = os.path.join(ranked_pages_dir, ranked_file_name)
        raw_file_path = os.path.join(raw_data_dir, f"{page_title}.txt")
        
        # Skip if raw document text is not available
        if not os.path.exists(raw_file_path):
            print(f"Warning: Raw text file for {page_title} not found at {raw_file_path}. Skipping.")
            continue
            
        # Load raw document text
        try:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
        except Exception as e:
            print(f"Error reading document text for {page_title}: {e}")
            continue
        
        # Load human-ranked data
        try:
            human_ranks_df = pd.read_csv(ranked_file_path)
            if not ({'expression', 'rank'}.issubset(human_ranks_df.columns)):
                print(f"Warning: CSV file {ranked_file_name} is missing required columns. Skipping.")
                continue
                
            # Process each expression
            for _, row in human_ranks_df.iterrows():
                expression = row['expression']
                human_rank = int(row['rank'])
                
                all_data.append({
                    'expression': expression,
                    'document_text': document_text,
                    'human_rank': human_rank,
                    'page_title': page_title
                })
                
        except Exception as e:
            print(f"Error processing human ranks for {page_title}: {e}")
            continue
    
    if not all_data:
        print("No training samples could be loaded. Ensure data exists and paths are correct.")
        return pd.DataFrame()
        
    return pd.DataFrame(all_data)

def train_ensemble_model(
    feature_model_path: str,
    encoder_model_path: str,
    ranked_pages_dir: str,
    raw_data_dir: str,
    stage_b_output_dir: str,
    output_dir: str = "models",
    output_name: str = "ensemble_ranker"
):
    """
    Train an ensemble model that combines feature-based and encoder-based models.
    
    Args:
        feature_model_path: Path to the feature-based model
        encoder_model_path: Path to the encoder-based model
        ranked_pages_dir: Directory with human-ranked CSVs
        raw_data_dir: Directory with raw text files
        stage_b_output_dir: Directory with Stage B similarity scores
        output_dir: Directory to save the ensemble model
        output_name: Name for the ensemble model files
    """
    print("Loading feature-based training data...")
    feature_df = load_training_data(
        ranked_pages_dir=ranked_pages_dir,
        raw_data_dir=raw_data_dir,
        stage_b_output_dir=stage_b_output_dir
    )
    
    if feature_df.empty:
        print("Error: No feature training data available")
        return
        
    print("Loading encoder training data...")
    encoder_df = load_encoder_data(
        ranked_pages_dir=ranked_pages_dir,
        raw_data_dir=raw_data_dir
    )
    
    if encoder_df.empty:
        print("Error: No encoder training data available")
        return
    
    # Merge dataframes to get common examples
    common_df = pd.merge(
        feature_df[['expression', 'page_title', 'human_rank']],
        encoder_df[['expression', 'page_title', 'document_text', 'human_rank']],
        on=['expression', 'page_title', 'human_rank'],
        how='inner'
    )
    
    print(f"Found {len(common_df)} common examples across both datasets")
    
    if len(common_df) < 10:
        print("Error: Not enough common examples to train ensemble")
        return
    
    # Extract data for training
    document_texts = common_df['document_text'].tolist()
    expressions = common_df['expression'].tolist()
    ranks = common_df['human_rank'].tolist()
    
    # Create similarity scores dictionary
    similarity_scores = {}
    for _, row in feature_df.iterrows():
        expr = row['expression']
        page = row['page_title']
        
        # Load similarity scores from Stage B if available
        stage_b_file = os.path.join(stage_b_output_dir, f"{page}_filtered.json")
        if os.path.exists(stage_b_file):
            try:
                with open(stage_b_file, 'r') as f:
                    page_scores = json.load(f)
                    if expr in page_scores:
                        similarity_scores[expr] = page_scores[expr]
            except Exception as e:
                print(f"Error loading similarity scores for {page}: {e}")
    
    # Split into train/test
    train_indices, test_indices = train_test_split(
        range(len(expressions)),
        test_size=0.2,
        random_state=42,
        stratify=ranks
    )
    
    train_docs = [document_texts[i] for i in train_indices]
    train_exprs = [expressions[i] for i in train_indices]
    train_ranks = [ranks[i] for i in train_indices]
    
    test_docs = [document_texts[i] for i in test_indices]
    test_exprs = [expressions[i] for i in test_indices]
    test_ranks = [ranks[i] for i in test_indices]
    
    # Create and train ensemble model
    print("\nInitializing ensemble model...")
    ensemble = EnsemblePrerequisiteRanker(
        feature_model_path=feature_model_path,
        encoder_model_path=encoder_model_path
    )
    
    print("\nTraining ensemble meta-learner...")
    ensemble.train(train_docs, train_exprs, train_ranks, similarity_scores)
    
    # Evaluate on test set
    print("\nEvaluating ensemble model...")
    all_predictions = {}
    all_true_ranks = {}
    
    for i in range(len(test_docs)):
        doc = test_docs[i]
        expr = test_exprs[i]
        true_rank = test_ranks[i]
        
        # Get prediction for single expression
        pred = ensemble.predict(
            doc, [expr], 
            {expr: similarity_scores.get(expr, 0.0)} if expr in similarity_scores else None
        )
        
        if pred:
            all_predictions[expr] = pred[expr]
            all_true_ranks[expr] = true_rank
    
    # Convert to lists for sklearn metrics
    y_true = list(all_true_ranks.values())
    y_pred = list(all_predictions.values())
    
    # Print evaluation metrics
    print("\nEnsemble Model Test Performance:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[0, 1, 2, 3],
               yticklabels=[0, 1, 2, 3])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Ensemble Model Confusion Matrix')
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/ensemble_confusion_matrix.png')
    print("\nConfusion matrix saved to 'results/ensemble_confusion_matrix.png'")
    
    # Save the ensemble model
    ensemble_output_dir = os.path.join(output_dir, output_name)
    ensemble.save(ensemble_output_dir)
    print(f"\nEnsemble model saved to {ensemble_output_dir}")

def main():
    """Main function to train the ensemble model"""
    # Configuration
    ranked_pages_dir = "data/raw/ranked_pages"
    raw_data_dir = "data/raw"
    stage_b_output_dir = "data/processed/stage_b"
    feature_model_path = "models/stage_c_ranker.joblib"
    encoder_model_path = "models/encoder_ranker.pt"
    output_dir = "models"
    
    print("Starting ensemble model training...")
    
    # Check if models exist
    if not os.path.exists(feature_model_path):
        print(f"Error: Feature model not found at {feature_model_path}")
        print("Please run train_ranker.py first.")
        return
        
    if not os.path.exists(encoder_model_path):
        print(f"Error: Encoder model not found at {encoder_model_path}")
        print("Please run train_ranker_encoder.py first.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the ensemble model
    train_ensemble_model(
        feature_model_path=feature_model_path,
        encoder_model_path=encoder_model_path,
        ranked_pages_dir=ranked_pages_dir,
        raw_data_dir=raw_data_dir,
        stage_b_output_dir=stage_b_output_dir,
        output_dir=output_dir,
        output_name="ensemble_ranker"
    )
    
    print("\nEnsemble model training complete!")

if __name__ == "__main__":
    main()
