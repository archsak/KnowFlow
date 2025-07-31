import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm
import sys
import random

# Add src to path for imports
sys.path.append('src')
sys.path.append('src/util')
from src.util.get_raw_text import get_raw_text

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset and Model
class PrerequisiteDataset(Dataset):
    """Dataset for prerequisite ranking using a transformer encoder."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512, article_texts: dict = None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        if article_texts:
            self.article_texts = article_texts
        else:
            print("Caching article texts")
            self.article_texts = {title: get_raw_text(title) for title in df['page_title'].unique()}
            print(f"Cached {len(self.article_texts)} article texts.")

        # For each row, create a sample for the start of the doc only (first 1000 characters)
        for i, row in df.iterrows():
            expression = str(row['expression'])
            page_title = str(row['page_title'])
            score = row['human_rank']
            document_text = self.article_texts[page_title]

            # Only use the first 1000 characters of the document as context
            doc_snippet = document_text[:1000] if len(document_text) > 1000 else document_text
            self.samples.append({
                'expression': expression,
                'page_title': page_title,
                'context': doc_snippet,
                'score': score
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        expression = sample['expression']
        page_title = sample['page_title']
        context = sample['context']
        score = sample['score']

        input_text = f"Rate prerequisite importance (0-3): Is '{expression}' essential before reading '{page_title}'? Context: {context}"
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float)
        }

class PrerequisiteRankerModel(nn.Module):
    """Enhanced transformer-based ranking model with better class separation."""
    
    def __init__(self, model_name="bert-base-uncased", dropout_rate=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Enhanced architecture with more layers and batch normalization
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Add sigmoid to ensure output is in [0,3] range
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's representation for the regression task
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.regressor(cls_output).squeeze(-1)
        # Scale sigmoid output to [0, 3] range
        return self.sigmoid(logits) * 3.0

# Soft Penalty Matrix Loss
class SoftPenaltyMatrixLoss(nn.Module):
    """
    Enhanced penalty matrix loss with additional MSE regularization 
    to encourage better class separation.
    """
    def __init__(self, penalty_matrix: torch.Tensor, mse_weight: float = 0.3):
        super().__init__()
        self.penalty_matrix = penalty_matrix
        self.num_classes = penalty_matrix.size(0)
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.penalty_matrix = self.penalty_matrix.to(targets.device)
        
        # Clamp predictions to valid range
        clamped_preds = torch.clamp(predictions, 0, self.num_classes - 1)
        target_classes = targets.long()
        
        # Calculate weights for interpolation
        lower_class = torch.floor(clamped_preds).long()
        upper_class = torch.ceil(clamped_preds).long()
        weight_upper = clamped_preds - lower_class.float()
        weight_lower = 1.0 - weight_upper
        
        # Get penalties for lower and upper classes
        penalties_lower = self.penalty_matrix[target_classes, lower_class]
        penalties_upper = self.penalty_matrix[target_classes, upper_class]
        
        # Interpolate penalties
        interpolated_penalties = weight_lower * penalties_lower + weight_upper * penalties_upper
        penalty_loss = interpolated_penalties.mean()
        
        # Add MSE loss to encourage better regression
        mse_loss = self.mse_loss(predictions, targets.float())
        
        # Combine losses
        total_loss = penalty_loss + self.mse_weight * mse_loss
        
        return total_loss

# Data Loading and Splitting
from sklearn.utils import resample

def load_training_data(csv_path="rated_wiki_pages.csv", oversample_2_3=True, group_concepts=True):
    """Loads data, groups similar concepts, and oversamples minority classes 2 and 3 more aggressively."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        'source_article': 'page_title',
        'concept': 'expression',
        'score': 'human_rank'
    })



    if oversample_2_3:
        # Separate by class
        df_0 = df[df['human_rank'] == 0]
        df_1 = df[df['human_rank'] == 1]
        df_2 = df[df['human_rank'] == 2]
        df_3 = df[df['human_rank'] == 3]

        target_size = max(len(df_1), len(df_0))  # Use the larger of classes 0 or 1
        
        # Oversample 2 and 3
        df_2_over = resample(df_2, replace=True, n_samples=target_size * 2, random_state=42)
        df_3_over = resample(df_3, replace=True, n_samples=target_size * 3, random_state=42)  # Extra emphasis on class 3

        # Combine all classes
        df_combined = pd.concat([df_0, df_1, df_2_over, df_3_over], ignore_index=True)
        
        print(f"Data distribution after oversampling:")
        print(f"Class 0: {len(df_0)}, Class 1: {len(df_1)}, Class 2: {len(df_2_over)}, Class 3: {len(df_3_over)}")
    else:
        df_combined = df

    return df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Model Evaluation
def detailed_prediction_analysis(predictions: np.ndarray, labels: np.ndarray, model_name: str, penalty_matrix: np.ndarray = None):
    """Provides a detailed analysis of prediction results and rounding methods."""
    print(f"\n {model_name}: Detailed Test Set Analysis")
    
    # Continuous prediction stats
    print(f"Continuous Prediction Stats: Min={predictions.min():.3f}, Max={predictions.max():.3f}, Mean={predictions.mean():.3f}")

    # Analyze different rounding methods
    methods = {'round': np.round, 'floor': np.floor, 'ceil': np.ceil}
    best_penalty = float('inf')
    best_method = ''
    
    print("\nRounding Method Comparison:")
    for name, method in methods.items():
        rounded_preds = np.clip(method(predictions).astype(int), 0, 3)
        accuracy = accuracy_score(labels, rounded_preds)
        mse = mean_squared_error(labels, rounded_preds)
        
        # Calculate penalty matrix cost if provided
        penalty_cost = 0
        if penalty_matrix is not None:
            for true_label, pred_label in zip(labels, rounded_preds):
                penalty_cost += penalty_matrix[int(true_label), int(pred_label)]
            penalty_cost /= len(labels)  # Average penalty
            
        print(f"Method: {name:5s} | Accuracy: {accuracy:.4f} | MSE: {mse:.4f} | Avg Penalty: {penalty_cost:.4f}")
        
        if penalty_matrix is not None and penalty_cost < best_penalty:
            best_penalty = penalty_cost
            best_method = name
        elif penalty_matrix is None and accuracy > best_penalty:  # Using accuracy as fallback
            best_penalty = accuracy
            best_method = name
            
    print(f"\nBest rounding method: '{best_method}' with {'penalty' if penalty_matrix is not None else 'accuracy'}: {best_penalty:.4f}")
    
    # Final evaluation with the best method
    final_preds = np.clip(methods[best_method](predictions).astype(int), 0, 3)
    
    print("\nDistribution of Final Predictions vs. Labels:")
    pred_counts = pd.Series(final_preds).value_counts().sort_index()
    label_counts = pd.Series(labels).value_counts().sort_index()
    # Fixed float issue by casting to int
    dist_df = pd.DataFrame({'predictions': pred_counts, 'labels': label_counts}).fillna(0).astype(int)
    print(dist_df)
    
    # Add confusion matrix for detailed accuracy
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, final_preds, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(4)], columns=[f'Predicted {i}' for i in range(4)])
    print(cm_df)

    # Calculate and print accuracy for classes 2 and 3
    print("\nAccuracy for key classes:")
    for i in [2, 3]:
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        if total > 0:
            accuracy = correct / total
            print(f"Class {i}: Accuracy = {accuracy:.4f} ({correct}/{total} correct)")
        else:
            print(f"Class {i}: No samples in test set.")

    return best_method

def evaluate_model(model, dataloader, device):
    """Evaluates the model, returning predictions and labels for validation."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['score'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Return raw predictions and labels for aggregation
    return predictions, labels

# Main Training Logic
def predict_with_concept_grouping(model_path: str, expressions: list, article_title: str, tokenizer=None, device=None):
    """
    Predicts prerequisite rankings for expressions, grouping similar concepts together.
    
    Args:
        model_path: Path to the trained model
        expressions: List of expression strings to rank
        article_title: The target article title
        tokenizer: Pre-loaded tokenizer (optional)
        device: Device to run inference on (optional)
    
    Returns:
        Dict mapping normalized concepts to their predicted rankings
    """
    print(f"Predicting prerequisite rankings for {len(expressions)} expressions.")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Load the trained model
    model = PrerequisiteRankerModel("bert-base-uncased").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    article_text = get_raw_text(article_title)
    if not article_text:
        print(f"Could not fetch text for article: {article_title}")
        return {}
    concept_rankings = {}
    with torch.no_grad():
        for expr in expressions:
            doc_snippet = article_text[:1000] if len(article_text) > 1000 else article_text
            input_text = f"Rate prerequisite importance (0-3): Is '{expr}' essential before reading '{article_title}'? Context: {doc_snippet}"
            encoding = tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            predicted_rank = output.item()
            concept_rankings[expr] = {
                'predicted_rank': predicted_rank,
                'rounded_rank': round(predicted_rank)
            }
    print(f"Generated rankings for {len(concept_rankings)} expressions")
    return concept_rankings

def main():
    """Main training and evaluation function."""
    print(" Stage C: Encoder-based Prerequisite Ranker Training with Penalty Matrix")
    set_seed(42)
    
    # Configuration
    MODEL_NAME = "bert-base-uncased"
    CSV_PATH = "rated_wiki_pages.csv"
    MODEL_OUTPUT_PATH = "models/stage_c_ranker_encoder_penalty.pt"
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    N_SPLITS = 4 # Number of folds for cross-validation
    
    penalty_matrix = torch.tensor([
        [0.0, 3.0, 25.0, 100.0],   # True 0: Very high penalty for predicting as 3
        [5.0, 0.0, 15.0, 80.0],    # True 1: High penalty for predicting as 3  
        [15.0, 8.0, 0.0, 12.0],    # True 2: Moderate penalty for wrong predictions
        [100.0, 60.0, 20.0, 0.0]   # True 3: Extremely high penalty for missing prerequisites
    ], dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Penalty Matrix:\n{penalty_matrix}")
    
    # Data Preparation
    data = load_training_data(CSV_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Cache all article texts once
    print("Caching all article texts for cross-validation")
    all_article_texts = {title: get_raw_text(title) for title in data['page_title'].unique()}
    print(f"Cached {len(all_article_texts)} article texts.")

    unique_articles = data['page_title'].unique()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_predictions = []
    all_labels = []

    # Cross-Validation Loop
    for fold, (train_article_idx, val_article_idx) in enumerate(kf.split(unique_articles)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        # Split articles for this fold
        train_articles = unique_articles[train_article_idx]
        val_articles = unique_articles[val_article_idx]
        
        train_df = data[data['page_title'].isin(train_articles)]
        val_df = data[data['page_title'].isin(val_articles)]
        
        print(f"Train: {len(train_articles)} articles, {len(train_df)} samples")
        print(f"Validation: {len(val_articles)} articles, {len(val_df)} samples")

        # Create datasets and dataloaders for this fold
        train_dataset = PrerequisiteDataset(train_df, tokenizer, article_texts=all_article_texts)
        val_dataset = PrerequisiteDataset(val_df, tokenizer, article_texts=all_article_texts)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 3)
        
        # Re-initialize model and optimizer for each fold
        model = PrerequisiteRankerModel(MODEL_NAME).to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        # Add learning rate scheduler
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 10,  # 10% warmup
            num_training_steps=total_steps
        )
        
        # Use enhanced penalty matrix loss
        criterion = SoftPenaltyMatrixLoss(penalty_matrix, mse_weight=0.5)
        
        # Training loop for this fold
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['score'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Fold {fold+1}, Epoch {epoch+1} - Avg Penalty Loss: {avg_loss:.4f}")

        # Evaluate on the validation set for this fold
        fold_preds, fold_labels = evaluate_model(model, val_loader, device)
        all_predictions.extend(fold_preds)
        all_labels.extend(fold_labels)

    # Final Evaluation on Aggregated Results
    print("\nCross-Validation Complete: Final Aggregated Evaluation")
    penalty_matrix_np = penalty_matrix.numpy()
    detailed_prediction_analysis(np.array(all_predictions), np.array(all_labels), 
                                "Aggregated Encoder Model with Penalty Matrix", penalty_matrix_np)
    
    print("\nTraining final model on all data")
    full_dataset = PrerequisiteDataset(data, tokenizer, article_texts=all_article_texts)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model = PrerequisiteRankerModel(MODEL_NAME).to(device)
    optimizer = AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Add scheduler for final training
    total_steps = len(full_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    criterion = SoftPenaltyMatrixLoss(penalty_matrix, mse_weight=0.5)

    for epoch in range(EPOCHS):
        final_model.train()
        for batch in tqdm(full_loader, desc=f"Final Training Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['score'].to(device)
            outputs = final_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nFinal model saved to {MODEL_OUTPUT_PATH}")
    print("\nProcess complete")

if __name__ == "__main__":
    main()