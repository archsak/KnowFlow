import os
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
import time
import tqdm

class PrerequisiteDataset(Dataset):
    """Dataset for prerequisite expression ranking"""
    
    def __init__(self, expressions, document_texts, ranks, tokenizer, max_length=512):
        """
        Initialize dataset
        
        Args:
            expressions: List of expressions to rank
            document_texts: List of document texts corresponding to each expression
            ranks: List of importance ranks (0-3) for each expression
            tokenizer: Tokenizer to use for encoding inputs
            max_length: Maximum sequence length for tokenization
        """
        self.expressions = expressions
        self.document_texts = document_texts
        self.ranks = ranks
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        expression = self.expressions[idx]
        document_text = self.document_texts[idx]
        rank = self.ranks[idx]
        
        # Create input format "[CLS] expression [SEP] document_text [SEP]"
        # Limited by max_length - we sample a segment of document text if it's too long
        if len(document_text) > 5000:  # Arbitrary threshold to avoid too long texts
            # Keep the beginning and end of the document, which often contain important context
            begin_part = document_text[:2000]  # First 2000 chars
            end_part = document_text[-2000:]   # Last 2000 chars
            document_text = begin_part + "..." + end_part
        
        # Tokenize inputs
        encoding = self.tokenizer(
            expression,
            document_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to the required format
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'rank': torch.tensor(rank, dtype=torch.float)  # Changed to float for regression
        }

class PrerequisiteRankerModel(nn.Module):
    """Encoder-based model for ranking expressions"""
    
    def __init__(self, encoder_name, use_regression=True):
        """
        Initialize model
        
        Args:
            encoder_name: Name of the encoder model (e.g., "bert-base-uncased")
            use_regression: If True, use regression (output single value), else classification (4 classes)
        """
        super(PrerequisiteRankerModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.use_regression = use_regression
        
        # Get encoder's hidden size
        hidden_size = self.encoder.config.hidden_size
        
        # Regression or classification head
        if use_regression:
            self.regressor = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)  # Single output for regression
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 4)  # 4 classes for ranks 0-3
            )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass"""
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        # Use [CLS] token embedding for prediction
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Regression or classification
        if self.use_regression:
            output = self.regressor(cls_output)
            return output.squeeze(-1)  # Remove last dimension for regression
        else:
            logits = self.classifier(cls_output)
            return logits

def load_training_data(ranked_csv_path: str, 
                       raw_data_dir: str,
                       stage_b_output_dir: str = None) -> pd.DataFrame:
    """
    Loads human-ranked expressions from a single CSV file, raw text documents,
    and optional similarity scores to create training data.
    Uses the same format as the feature-based model.

    Args:
        ranked_csv_path: Path to the combined CSV file of human-ranked expressions.
        raw_data_dir: Directory containing raw text files (one per article).
        stage_b_output_dir: Optional directory containing Stage B similarity scores.

    Returns:
        A pandas DataFrame with expressions, document texts and labels.
    """
    all_training_samples = []

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

            # Add data in the same format as feature model
            all_training_samples.append({
                'expression': expression,
                'page_title': page_title,
                'document_text': document_text,
                'human_rank': human_rank,
                'similarity_score': similarity_score
            })

    if not all_training_samples:
        print("No training samples could be loaded. Ensure data exists and paths are correct.")
        return pd.DataFrame()

    loaded_df = pd.DataFrame(all_training_samples)
    loaded_df = loaded_df.fillna(-1)
    return loaded_df

def split_data_by_articles(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits data by articles to avoid data leakage.
    """
    articles = df['page_title'].unique()
    np.random.seed(42)
    np.random.shuffle(articles)
    
    n_articles = len(articles)
    train_end = int(n_articles * train_ratio)
    val_end = int(n_articles * (train_ratio + val_ratio))
    
    train_articles = articles[:train_end]
    val_articles = articles[train_end:val_end]
    test_articles = articles[val_end:]
    
    train_df = df[df['page_title'].isin(train_articles)]
    val_df = df[df['page_title'].isin(val_articles)]
    test_df = df[df['page_title'].isin(test_articles)]
    
    print(f"Data split by articles:")
    print(f"Train: {len(train_articles)} articles, {len(train_df)} samples")
    print(f"Validation: {len(val_articles)} articles, {len(val_df)} samples")  
    print(f"Test: {len(test_articles)} articles, {len(test_df)} samples")
    
    return train_df, val_df, test_df


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model and return MSE and accuracy scores.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['score'].to(device)  # Changed from 'rank' to 'score'
            
            outputs = model(input_ids, attention_mask)
            
            predictions = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels_np)
    
    # Round predictions for accuracy calculation
    rounded_predictions = np.round(np.array(all_predictions)).astype(int)
    rounded_predictions = np.clip(rounded_predictions, 0, 3)  # Clamp to valid range
    
    # Calculate MSE and accuracy
    mse = mean_squared_error(all_labels, rounded_predictions)
    accuracy = accuracy_score(all_labels, rounded_predictions)
    
    return mse, accuracy


def load_training_data_simple(ranked_csv_path: str = "data/raw/ranked_pages/rated_wiki_pages.csv") -> pd.DataFrame:
    """
    Simple load function that loads data in the same format as the feature model.
    """
    if not os.path.exists(ranked_csv_path):
        print(f"Error: Ranked CSV not found at {ranked_csv_path}")
        return pd.DataFrame()

    try:
        ranked_df = pd.read_csv(ranked_csv_path)
    except Exception as e:
        print(f"Failed to load ranked CSV: {e}")
        return pd.DataFrame()

    # Rename columns to match expected format
    if 'source_article' in ranked_df.columns:
        ranked_df = ranked_df.rename(columns={
            'source_article': 'page_title',
            'concept': 'expression', 
            'score': 'human_rank'
        })

    # Validate expected columns
    if not {'page_title', 'expression', 'human_rank'}.issubset(ranked_df.columns):
        print("Error: CSV must contain 'page_title', 'expression', and 'human_rank' columns.")
        return pd.DataFrame()

    print(f"Loaded {len(ranked_df)} samples from {len(ranked_df['page_title'].unique())} articles")
    return ranked_df


class SimplePrerequisiteDataset(Dataset):
    """Simplified dataset that works directly with the CSV data."""
    
    def __init__(self, expressions, page_titles, scores, max_length=512):
        self.expressions = expressions
        self.page_titles = page_titles
        self.scores = scores
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.expressions)
        
    def __getitem__(self, idx):
        expression = str(self.expressions[idx])
        page_title = str(self.page_titles[idx])
        score = float(self.scores[idx])
        
        # Combine page title and expression for context
        text = f"Article: {page_title}. Concept: {expression}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'score': torch.tensor(score, dtype=torch.float)
        }


def train_encoder_model(
    training_data,
    encoder_name="bert-base-uncased",
    output_dir="models",
    model_output_name="encoder_ranker.pt",
    batch_size=16,
    epochs=4,
    learning_rate=2e-5,
    device=None,
    use_regression=True
):
    """
    Train an encoder-based ranking model
    
    Args:
        training_data: DataFrame with 'expression', 'document_text', and 'human_rank'
        encoder_name: Name of pretrained encoder model
        output_dir: Directory to save model
        model_output_name: Filename for saved model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        use_regression: If True, use regression, else classification
    """
    if training_data.empty:
        print("No training data provided.")
        return None
        
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    # Split into train and validation sets
    if use_regression:
        # For regression, we don't need stratification
        train_df, val_df = train_test_split(
            training_data,
            test_size=0.2,
            random_state=42
        )
    else:
        # For classification, use stratification
        train_df, val_df = train_test_split(
            training_data,
            test_size=0.2,
            random_state=42,
            stratify=training_data['human_rank']
        )
    
    # Print dataset statistics
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Target distribution in training: {train_df['human_rank'].value_counts().sort_index()}")
    
    # Create datasets
    train_dataset = PrerequisiteDataset(
        expressions=train_df['expression'].tolist(),
        document_texts=train_df['document_text'].tolist(),
        ranks=train_df['human_rank'].tolist(),
        tokenizer=tokenizer
    )
    
    val_dataset = PrerequisiteDataset(
        expressions=val_df['expression'].tolist(),
        document_texts=val_df['document_text'].tolist(),
        ranks=val_df['human_rank'].tolist(),
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    print(f"Initializing model with {encoder_name}")
    model = PrerequisiteRankerModel(encoder_name=encoder_name, use_regression=use_regression)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * epochs
    
    # Create scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function
    if use_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm.tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            labels = batch['rank'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if 'token_type_ids' in batch else None
            )
            
            # Compute loss
            if use_regression:
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels.long())
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix(loss=train_loss / (progress_bar.n + 1))
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
                labels = batch['rank'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids if 'token_type_ids' in batch else None
                )
                
                if use_regression:
                    loss = criterion(outputs, labels)
                    # Round predictions to nearest integer for evaluation
                    preds = torch.round(outputs).clamp(0, 3)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    loss = criterion(outputs, labels.long())
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Convert predictions to integers for evaluation
        if use_regression:
            all_preds_int = [int(round(p)) for p in all_preds]
            all_labels_int = [int(round(l)) for l in all_labels]
            from sklearn.metrics import mean_squared_error, accuracy_score
            mse = mean_squared_error(all_labels_int, all_preds_int)
            accuracy = accuracy_score(all_labels_int, all_preds_int)
            print(f"MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")
        else:
            print(classification_report(all_labels, all_preds, zero_division=0))
            
        cm = confusion_matrix(all_labels_int if use_regression else all_labels, 
                            all_preds_int if use_regression else all_preds)
        print("Confusion Matrix:")
        print(cm)
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(os.path.join(output_dir, "encoder_confusion_matrix.csv"), cm, delimiter=",", fmt='%d')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model state.")
    
    # Save final model
    output_model_path = os.path.join(output_dir, model_output_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_name': encoder_name,
        'use_regression': use_regression
    }, output_model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    
    print(f"Model saved to {output_model_path}")
    print(f"Tokenizer saved to {os.path.join(output_dir, 'tokenizer')}")
    
    return model

def train_and_save_model(model_output_path: str, epochs: int = 2, device: str = None):
    """Train the BERT encoder model using the rating data."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    data = load_training_data_simple()
    
    if data.empty:
        print("No training data loaded. Exiting.")
        return None
    
    print(f"Loaded {len(data)} samples from {len(data['page_title'].unique())} articles")
    
    # Split data by articles to avoid data leakage
    train_df, val_df, test_df = split_data_by_articles(data)
    
    # Create datasets and dataloaders
    train_dataset = SimplePrerequisiteDataset(
        train_df['expression'].tolist(),
        train_df['page_title'].tolist(), 
        train_df['human_rank'].tolist()
    )
    
    val_dataset = SimplePrerequisiteDataset(
        val_df['expression'].tolist(),
        val_df['page_title'].tolist(),
        val_df['human_rank'].tolist()
    )
    
    test_dataset = SimplePrerequisiteDataset(
        test_df['expression'].tolist(),
        test_df['page_title'].tolist(),
        test_df['human_rank'].tolist()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Smaller batch size
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    print("Initializing BERT model...")
    model = PrerequisiteRankerModel(encoder_name="bert-base-uncased", use_regression=True)
    model = model.to(device)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {epochs} epochs...")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['score'].float().to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
        
        # Validation evaluation
        if epoch % 1 == 0:  # Evaluate every epoch
            model.eval()
            val_mse, val_accuracy = evaluate_model(model, val_loader, device)
            print(f"Validation MSE: {val_mse:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            model.train()
    
    # Final test evaluation
    print("Running final test evaluation...")
    model.eval()
    test_mse, test_accuracy = evaluate_model(model, test_loader, device)
    print(f"\nFinal Test Results:")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"\nTrained encoder model saved to {model_output_path}")
    
    return model


def evaluate_model_detailed(model, test_data, tokenizer, device, use_regression=True):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained model
        test_data: DataFrame with 'expression', 'document_text', and 'human_rank'
        tokenizer: Tokenizer used for encoding inputs
        device: Device to evaluate on ('cuda' or 'cpu')
        use_regression: Whether the model uses regression or classification
    
    Returns:
        DataFrame with test expressions, predicted ranks, and human ranks
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            expression = row['expression']
            document_text = row['document_text']
            human_rank = row['human_rank']
            
            # Tokenize inputs
            encoding = tokenizer(
                expression,
                document_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze().to(device)
            attention_mask = encoding['attention_mask'].squeeze().to(device)
            token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).squeeze().to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                token_type_ids=token_type_ids.unsqueeze(0) if token_type_ids is not None else None
            )
            
            if use_regression:
                # Round to nearest integer and clamp to valid range
                pred = torch.round(outputs).clamp(0, 3).cpu().numpy()[0]
            else:
                pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
                
            all_preds.append(pred)
            all_labels.append(human_rank)
    
    return pd.DataFrame({
        'expression': test_data['expression'],
        'document_text': test_data['document_text'],
        'predicted_rank': all_preds,
        'human_rank': all_labels
    })

def main():
    """
    Main function for training the encoder-based ranking model.
    Uses the same data format and paths as the feature-based model.
    """
    # Paths (same as feature-based model)
    ranked_csv_path = "data/raw/ranked_pages/rated_wiki_pages.csv"
    raw_data_dir = "data/raw/raw_texts"
    stage_b_output_dir = "data/processed/stage_b"  # Optional
    model_output_path = "models/stage_c_ranker_encoder.pt"
    
    print("=== Stage C: Encoder-based Ranking Model Training ===")
    print(f"Loading training data from {ranked_csv_path}")
    print(f"Raw documents directory: {raw_data_dir}")
    print(f"Stage B output directory: {stage_b_output_dir}")
    print(f"Model output path: {model_output_path}")

    # Load training data (same function as feature-based model)
    training_df = load_training_data(
        ranked_csv_path=ranked_csv_path,
        raw_data_dir=raw_data_dir,
        stage_b_output_dir=stage_b_output_dir
    )

    if training_df.empty:
        print("Failed to load training data. Exiting.")
        return

    print(f"\nLoaded {len(training_df)} training samples.")
    print(f"Rank distribution in loaded data:\n{training_df['human_rank'].value_counts().sort_index()}")
    
    # Show a few examples
    print("\nSample data (first few rows with selected columns):")
    sample_columns = ['expression', 'page_title', 'human_rank']
    available_columns = [col for col in sample_columns if col in training_df.columns]
    print(training_df[available_columns].head())

    # Train and save the encoder model
    print("\nStarting encoder model training...")
    model = train_encoder_model(
        training_data=training_df,
        encoder_name="bert-base-uncased",
        output_dir="models",
        model_output_name="stage_c_ranker_encoder.pt",
        batch_size=16,
        epochs=4,
        learning_rate=2e-5,
        use_regression=True  # Use regression like the feature-based model
    )
    
    print("\nStage C encoder model training finished.")

if __name__ == "__main__":
    main()
