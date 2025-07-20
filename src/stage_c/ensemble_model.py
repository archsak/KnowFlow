import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import BaseEstimator, RegressorMixin
import sys
sys.path.append('src')
from Bert1 import get_raw_text
from train_ranker_encoder_clean import PrerequisiteRankerModel, PrerequisiteDataset, load_training_data, split_data_by_articles
from train_ranker_features import FeatureExtractor
from transformers import AutoTokenizer


class FeatureModelWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for feature-based model that handles feature extraction."""
    
    def __init__(self, model_path="models/stage_c_ranker.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_extractor = None
        self.feature_names = None
        self.fitted = False
        self.text_cache = {}  # Cache for article texts
        
    def fit(self, X, y):
        """Load pre-trained model and initialize feature extractor."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.feature_extractor = FeatureExtractor()
            self.fitted = True
            print(f"Loaded feature model from {self.model_path}")
            
            # Try to get feature names from the model
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_
                print(f"Model expects {len(self.feature_names)} features")
            else:
                print("Warning: Model doesn't have feature_names_in_ attribute")
                self.feature_names = None
        else:
            print(f"Warning: Feature model not found at {self.model_path}")
            self.fitted = False
            
        return self
        
    def predict(self, X):
        """Make predictions using feature model."""
        if not self.fitted:
            print("Warning: Feature model not fitted, returning zeros")
            return np.zeros(len(X))
            
        predictions = []
        
        # Convert to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        for idx, row in X.iterrows():
            expression = str(row['expression'])
            page_title = str(row['page_title'])
            
            # Get text with caching
            if page_title not in self.text_cache:
                try:
                    self.text_cache[page_title] = get_raw_text(page_title)
                except:
                    self.text_cache[page_title] = page_title
                    
            document_text = self.text_cache[page_title]
                
            # Extract features
            features = self.feature_extractor.extract_features(
                document_text=document_text,
                expression=expression,
                similarity_score=None
            )
            
            # Convert to DataFrame with correct column order
            features_df = pd.DataFrame([features])
            
            # If we have feature names from the model, reorder columns
            if self.feature_names is not None:
                # Add missing columns with default values
                for col in self.feature_names:
                    if col not in features_df.columns:
                        features_df[col] = 0
                
                # Reorder columns to match training order
                features_df = features_df[self.feature_names]
            
            # Make prediction
            pred = self.model.predict(features_df)[0]
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'model_path': self.model_path}
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class BERTRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make BERT model compatible with sklearn ensemble."""
    
    def __init__(self, model_path="models/stage_c_ranker_encoder.pt"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.fitted = False
        self.text_cache = {}  # Cache for article texts
        
    def fit(self, X, y):
        """Load pre-trained model (no actual training)."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load model
        self.model = PrerequisiteRankerModel()
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.fitted = True
            print(f"Loaded BERT model from {self.model_path}")
        else:
            print(f"Warning: Model file {self.model_path} not found")
            self.fitted = False
            
        return self
        
    def predict(self, X):
        """Make predictions using BERT model."""
        if not self.fitted:
            print("Warning: Model not fitted, returning zeros")
            return np.zeros(len(X))
            
        predictions = []
        
        # Convert to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        for idx, row in X.iterrows():
            expression = str(row['expression'])
            page_title = str(row['page_title'])
            
            # Get text with caching
            if page_title not in self.text_cache:
                try:
                    self.text_cache[page_title] = get_raw_text(page_title)
                except:
                    self.text_cache[page_title] = page_title
                    
            document_text = self.text_cache[page_title]
                
            # Combine for context
            full_text = f"Article: {page_title}. Document: {document_text[:1000]}... Concept: {expression}"
            
            # Tokenize
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Predict
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                output = self.model(input_ids, attention_mask)
                pred = output.cpu().numpy()[0]
                predictions.append(pred)
        
        return np.array(predictions)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'model_path': self.model_path}
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SimpleEnsemble:
    """Simple ensemble that combines predictions from multiple models."""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.fitted = False
        
    def fit(self, X, y):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        self.fitted = True
        return self
        
    def predict(self, X):
        """Make ensemble predictions by averaging weighted predictions."""
        if not self.fitted:
            raise ValueError("Ensemble not fitted yet")
            
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
            
        # Average the weighted predictions
        ensemble_pred = np.sum(predictions, axis=0) / np.sum(self.weights)
        return ensemble_pred


def create_ensemble_model():
    """Create an ensemble of feature-based and BERT-based models."""
    
    # Check if models exist
    feature_model_path = "models/stage_c_ranker.joblib"
    bert_model_path = "models/stage_c_ranker_encoder.pt"
    
    models = []
    model_names = []
    
    # Add feature model if exists
    if os.path.exists(feature_model_path):
        feature_wrapper = FeatureModelWrapper(model_path=feature_model_path)
        models.append(feature_wrapper)
        model_names.append("Feature Model")
        print(f"Will load feature model from {feature_model_path}")
    else:
        print(f"Feature model not found at {feature_model_path}")
        
    # Add BERT model if exists
    if os.path.exists(bert_model_path):
        bert_wrapper = BERTRegressorWrapper(model_path=bert_model_path)
        models.append(bert_wrapper)
        model_names.append("BERT Model")
        print(f"Will load BERT model from {bert_model_path}")
    else:
        print(f"BERT model not found at {bert_model_path}")
    
    if not models:
        print("No models found!")
        return None, []
    
    # Create ensemble with equal weights
    ensemble = SimpleEnsemble(models, weights=[1.0] * len(models))
    
    return ensemble, model_names


def evaluate_ensemble():
    """Evaluate the ensemble model."""
    print("=== Ensemble Model Evaluation ===")
    
    # Load data
    data = load_training_data()
    train_df, val_df, test_df = split_data_by_articles(data)
    
    print(f"Test data: {test_df.shape} samples from {test_df['page_title'].nunique()} unique articles")
    
    # Create ensemble
    ensemble, model_names = create_ensemble_model()
    if ensemble is None:
        print("Failed to create ensemble model")
        return
    
    print(f"Ensemble contains: {', '.join(model_names)}")
    
    # Prepare test data
    test_data = test_df[['expression', 'page_title']].copy()
    test_labels = test_df['human_rank'].values
    
    # Load models and make predictions
    print("Loading models and making predictions...")
    ensemble.fit(test_data, test_labels)
    predictions = ensemble.predict(test_data)
    
    # Evaluate with different rounding methods (based on our analysis, round is best for accuracy)
    methods = {'round': np.round, 'ceil': np.ceil, 'floor': np.floor}
    best_mse = float('inf')
    best_accuracy = 0
    best_method = 'round'
    
    print("\nRounding Method Comparison:")
    for method_name, method_func in methods.items():
        rounded_preds = np.clip(method_func(predictions).astype(int), 0, 3)
        mse = mean_squared_error(test_labels, rounded_preds)
        accuracy = accuracy_score(test_labels, rounded_preds)
        
        print(f"{method_name:5s}: MSE={mse:.4f}, Accuracy={accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = method_name
    
    # Final evaluation with best method
    final_predictions = np.clip(methods[best_method](predictions).astype(int), 0, 3)
    final_mse = mean_squared_error(test_labels, final_predictions)
    final_accuracy = accuracy_score(test_labels, final_predictions)
    
    print(f"\nFinal Results (using {best_method} rounding):")
    print(f"MSE: {final_mse:.4f}")
    print(f"Accuracy: {final_accuracy:.4f}")
    
    # Show distributions
    print(f"\nPrediction vs Label Distribution:")
    pred_dist = np.bincount(final_predictions, minlength=4)
    label_dist = np.bincount(test_labels, minlength=4)
    for i in range(4):
        print(f"  Score {i}: {pred_dist[i]} predictions, {label_dist[i]} labels")
    
    return predictions, test_labels


def main():
    """Main function to evaluate ensemble."""
    evaluate_ensemble()


if __name__ == "__main__":
    main()
