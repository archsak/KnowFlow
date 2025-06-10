import os
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
            'rank': torch.tensor(rank, dtype=torch.long)
        }

class PrerequisiteRankerModel(nn.Module):
    """Encoder-based model for ranking expressions"""
    
    def __init__(self, encoder_name, num_classes=4):  # 4 classes for ranks 0-3
        """
        Initialize model
        
        Args:
            encoder_name: Name of the encoder model (e.g., "bert-base-uncased")
            num_classes: Number of output classes
        """
        super(PrerequisiteRankerModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Get encoder's hidden size
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass"""
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        
        # Use [CLS] token embedding for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(cls_output)
        
        return logits

def load_training_data(ranked_pages_dir, raw_data_dir, stage_b_output_dir=None):
    """
    Load training data from human-ranked expressions and document texts
    
    Args:
        ranked_pages_dir: Directory containing CSV files with human-ranked expressions
        raw_data_dir: Directory containing raw document text files
        stage_b_output_dir: Optional directory with Stage B scores
    
    Returns:
        DataFrame containing expressions, document texts, and ranks
    """
    all_data = []
    
    if not os.path.exists(ranked_pages_dir):
        print(f"Error: Ranked pages directory not found: {ranked_pages_dir}")
        return pd.DataFrame()
    
    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        return pd.DataFrame()
        
    for file_name in os.listdir(ranked_pages_dir):
        if not file_name.endswith('.csv'):
            continue
            
        page_title = file_name.replace('.csv', '')
        ranked_file_path = os.path.join(ranked_pages_dir, file_name)
        raw_file_path = os.path.join(raw_data_dir, f"{page_title}.txt")
        
        if not os.path.exists(raw_file_path):
            print(f"Warning: Raw text not found for page {page_title}. Skipping.")
            continue
            
        # Load document text
        try:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
        except Exception as e:
            print(f"Error reading document {page_title}: {e}")
            continue
            
        # Load human ranks
        try:
            ranks_df = pd.read_csv(ranked_file_path)
            if not {'expression', 'rank'}.issubset(ranks_df.columns):
                print(f"Warning: CSV file {file_name} missing required columns. Skipping.")
                continue
                
            # Process each expression
            for _, row in ranks_df.iterrows():
                expression = row['expression']
                rank = int(row['rank'])
                
                all_data.append({
                    'page_title': page_title,
                    'expression': expression,
                    'document_text': document_text,
                    'human_rank': rank
                })
                
        except Exception as e:
            print(f"Error processing ranks for {page_title}: {e}")
            continue
    
    if not all_data:
        print("No training samples could be loaded.")
        return pd.DataFrame()
        
    return pd.DataFrame(all_data)

def train_encoder_model(
    training_data,
    encoder_name="bert-base-uncased",
    output_dir="models",
    model_output_name="encoder_ranker.pt",
    batch_size=16,
    epochs=4,
    learning_rate=2e-5,
    device=None
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
    train_df, val_df = train_test_split(
        training_data,
        test_size=0.2,
        random_state=42,
        stratify=training_data['human_rank']
    )
    
    # Print dataset statistics
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Class distribution in training: {train_df['human_rank'].value_counts().sort_index()}")
    
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
    model = PrerequisiteRankerModel(encoder_name=encoder_name)
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
            loss = criterion(outputs, labels)
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
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(classification_report(all_labels, all_preds, zero_division=0))
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)
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
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
    
    return model

def evaluate_model(model, test_data, tokenizer, device):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained model
        test_data: DataFrame with 'expression', 'document_text', and 'human_rank'
        tokenizer: Tokenizer used for encoding inputs
        device: Device to evaluate on ('cuda' or 'cpu')
    
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
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy()[0])
            all_labels.append(human_rank)
    
    return pd.DataFrame({
        'expression': test_data['expression'],
        'document_text': test_data['document_text'],
        'predicted_rank': all_preds,
        'human_rank': all_labels
    })
