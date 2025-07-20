import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, accuracy_score
from tqdm import tqdm
import sys
import random

# Add src to path for imports
sys.path.append('src')
from stage_a.Bert1 import get_raw_text

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
    
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's representation for the regression task
        cls_output = outputs.last_hidden_state[:, 0]
        return self.regressor(cls_output).squeeze(-1)

# --- Data Loading and Splitting ---
def load_training_data(csv_path="data/raw/ranked_pages/rated_wiki_pages.csv"):
    """Loads and renames columns from the training data CSV."""
    df = pd.read_csv(csv_path)
    return df.rename(columns={
        'source_article': 'page_title',
        'concept': 'expression', 
        'score': 'human_rank'
    })

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
    dist_df = pd.DataFrame({'predictions': pred_counts, 'labels': label_counts}).fillna(0)
    print(dist_df)
    
    return best_method

def evaluate_model(model, dataloader, device, is_test_set=False):
    """Evaluates the model, returning predictions and labels."""
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
    
    if is_test_set:
        detailed_prediction_analysis(predictions, labels, "Encoder-based Model")
    
    # For validation, just return standard metrics
    rounded_predictions = np.clip(np.round(predictions).astype(int), 0, 3)
    mse = mean_squared_error(labels, rounded_predictions)
    accuracy = accuracy_score(labels, rounded_predictions)
    
    return mse, accuracy, predictions, labels

# --- Main Training Logic ---
def main():
    """Main training and evaluation function."""
    print("=== Stage C: Encoder-based Prerequisite Ranker Training ===")
    set_seed(42)
    
    # --- Configuration ---
    MODEL_NAME = "bert-base-uncased"
    CSV_PATH = "data/raw/ranked_pages/rated_wiki_pages.csv"
    MODEL_OUTPUT_PATH = "models/stage_c_ranker_encoder.pt"
    EPOCHS = 3
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Data Preparation ---
    data = load_training_data(CSV_PATH)
    train_df, val_df, test_df = split_data_by_articles(data)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = PrerequisiteDataset(train_df, tokenizer)
    val_dataset = PrerequisiteDataset(val_df, tokenizer)
    test_dataset = PrerequisiteDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)
    
    # --- Model Training ---
    model = PrerequisiteRankerModel(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print(f"\nStarting training for {EPOCHS} epochs...")
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
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Validation
        val_mse, val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Validation - MSE: {val_mse:.4f}, Accuracy: {val_accuracy:.4f}")
    
    # --- Final Evaluation ---
    print("\n--- Final Test Set Evaluation ---")
    evaluate_model(model, test_loader, device, is_test_set=True)
    
    # --- Save Model ---
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
    print("\n--- Training complete ---")

if __name__ == "__main__":
    main()
