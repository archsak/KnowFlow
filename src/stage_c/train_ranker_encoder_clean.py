import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm
import sys
import random

# Add src to path for imports
sys.path.append('src')
from Bert1 import get_raw_text

# --- Utility Functions ---
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

# --- Dataset and Model ---
class PrerequisiteDataset(Dataset):
    """Dataset for prerequisite ranking using a transformer encoder."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512, article_texts: dict = None):
        self.expressions = df['expression'].tolist()
        self.page_titles = df['page_title'].tolist()
        self.scores = df['human_rank'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if article_texts:
            self.article_texts = article_texts
        else:
            print("Caching article texts...")
            self.article_texts = {title: get_raw_text(title) for title in df['page_title'].unique()}
            print(f"Cached {len(self.article_texts)} article texts.")
        
    def __len__(self):
        return len(self.expressions)
        
    def __getitem__(self, idx):
        expression = str(self.expressions[idx])
        page_title = str(self.page_titles[idx])
        document_text = self.article_texts[page_title]
        
        # Create a focused input text
        input_text = f"Concept: {expression} [SEP] Article: {page_title} [SEP] {document_text}"
        
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
            'score': torch.tensor(self.scores[idx], dtype=torch.float)
        }

class PrerequisiteRankerModel(nn.Module):
    """Transformer-based ranking model with a regression head."""
    
    def __init__(self, model_name="bert-base-uncased", dropout_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's representation for the regression task
        cls_output = outputs.last_hidden_state[:, 0]
        return self.regressor(cls_output).squeeze(-1)

# --- Custom Penalty Matrix Loss Function ---
class PenaltyMatrixLoss(nn.Module):
    """
    Loss function based on a penalty matrix that defines different penalties 
    for different prediction-target combinations.
    """
    def __init__(self, penalty_matrix: torch.Tensor):
        """
        Args:
            penalty_matrix (torch.Tensor): A 2D tensor where penalty_matrix[i, j] 
                                         is the penalty for predicting class j when 
                                         the true class is i.
        """
        super().__init__()
        self.penalty_matrix = penalty_matrix
        self.num_classes = penalty_matrix.size(0)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the penalty matrix-based loss.
        
        Args:
            predictions (torch.Tensor): The model's continuous predictions
            targets (torch.Tensor): The ground truth labels (integers 0-3)
            
        Returns:
            torch.Tensor: The calculated penalty-based loss
        """
        # Move penalty matrix to the same device as the targets
        self.penalty_matrix = self.penalty_matrix.to(targets.device)
        
        # Round predictions to nearest integer and clamp to valid range
        pred_classes = torch.round(predictions).clamp(0, self.num_classes - 1).long()
        target_classes = targets.long()
        
        # Get penalties for each sample using advanced indexing
        penalties = self.penalty_matrix[target_classes, pred_classes]
        
        # Return mean penalty
        return penalties.mean()

# --- Alternative: Soft Penalty Matrix Loss (for continuous predictions) ---
class SoftPenaltyMatrixLoss(nn.Module):
    """
    Soft version of penalty matrix loss that interpolates penalties based on 
    continuous predictions instead of rounding.
    """
    def __init__(self, penalty_matrix: torch.Tensor):
        super().__init__()
        self.penalty_matrix = penalty_matrix
        self.num_classes = penalty_matrix.size(0)

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
        
        return interpolated_penalties.mean()

# --- Data Loading and Splitting ---
from sklearn.utils import resample

def load_training_data(csv_path="rated_wiki_pages.csv", oversample_2_3=True):
    """Loads data and optionally oversamples minority classes 2 and 3."""
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

        # Determine target size (e.g., match majority class - 1)
        target_size = len(df_1)

        # Oversample 2 and 3 to target_size
        df_2_over = resample(df_2, replace=True, n_samples=target_size, random_state=42)
        df_3_over = resample(df_3, replace=True, n_samples=target_size, random_state=42)

        # Optionally: keep 0 small or oversample as well (your call)
        df_combined = pd.concat([df_0, df_1, df_2_over, df_3_over], ignore_index=True)
    else:
        df_combined = df

    return df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Model Evaluation ---
def detailed_prediction_analysis(predictions: np.ndarray, labels: np.ndarray, model_name: str, penalty_matrix: np.ndarray = None):
    """Provides a detailed analysis of prediction results and rounding methods."""
    print(f"\n--- {model_name}: Detailed Test Set Analysis ---")
    
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

# --- Main Training Logic ---
def main():
    """Main training and evaluation function."""
    print("=== Stage C: Encoder-based Prerequisite Ranker Training with Penalty Matrix ===")
    set_seed(42)
    
    # --- Configuration ---
    MODEL_NAME = "bert-base-uncased"
    CSV_PATH = "rated_wiki_pages.csv"
    MODEL_OUTPUT_PATH = "models/stage_c_ranker_encoder_penalty.pt"
    EPOCHS = 3
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    N_SPLITS = 4 # Number of folds for cross-validation
    
    # Define penalty matrix
    penalty_matrix = torch.tensor([
        [0.0, 2.0, 15.0, 40.0],
        [3.0, 0.0, 12.0, 35.0],
        [8.0, 5.0, 0.0, 7.0],
        [40.0, 30.0, 12.0, 0.0]
    ], dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Penalty Matrix:\n{penalty_matrix}")
    
    # --- Data Preparation ---
    data = load_training_data(CSV_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Cache all article texts once
    print("Caching all article texts for cross-validation...")
    all_article_texts = {title: get_raw_text(title) for title in data['page_title'].unique()}
    print(f"Cached {len(all_article_texts)} article texts.")

    unique_articles = data['page_title'].unique()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_predictions = []
    all_labels = []

    # --- Cross-Validation Loop ---
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
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)
        
        # Re-initialize model and optimizer for each fold
        model = PrerequisiteRankerModel(MODEL_NAME).to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # Use penalty matrix loss instead of weighted MSE
        criterion = SoftPenaltyMatrixLoss(penalty_matrix)  # or PenaltyMatrixLoss for hard rounding
        
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
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Fold {fold+1}, Epoch {epoch+1} - Avg Penalty Loss: {avg_loss:.4f}")

        # Evaluate on the validation set for this fold
        fold_preds, fold_labels = evaluate_model(model, val_loader, device)
        all_predictions.extend(fold_preds)
        all_labels.extend(fold_labels)

    # --- Final Evaluation on Aggregated Results ---
    print("\n--- Cross-Validation Complete: Final Aggregated Evaluation ---")
    penalty_matrix_np = penalty_matrix.numpy()
    detailed_prediction_analysis(np.array(all_predictions), np.array(all_labels), 
                                "Aggregated Encoder Model with Penalty Matrix", penalty_matrix_np)
    
    # Optional: Train final model on all data and save it
    print("\n--- Training final model on all data ---")
    full_dataset = PrerequisiteDataset(data, tokenizer, article_texts=all_article_texts)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model = PrerequisiteRankerModel(MODEL_NAME).to(device)
    optimizer = AdamW(final_model.parameters(), lr=LEARNING_RATE)
    criterion = SoftPenaltyMatrixLoss(penalty_matrix)

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

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nFinal model saved to {MODEL_OUTPUT_PATH}")
    print("\n--- Process complete ---")

if __name__ == "__main__":
    main()