import os
import json
import torch
from torch import nn
from transformers import AutoTokenizer
import tqdm
from typing import List, Dict, Any, Tuple

class PrerequisiteRankerEncoder:
    """
    Ranks expressions based on their importance for understanding text
    using a pre-trained encoder-based deep learning model.
    Assigns importance scores from 0-3 using regression.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the ranker with a trained encoder model
        
        Args:
            model_path: Path to the saved encoder model (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
       
       
        try:
            # Set these explicitly since the checkpoint doesn't contain metadata
            self.encoder_name = 'bert-base-uncased'  # update if different
            self.use_regression = True

            # Load tokenizer
            tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                print(f"Tokenizer not found at {tokenizer_path}, loading from HuggingFace")
                self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)

            # Initialize model architecture (same as training script)
            from transformers import AutoModel

            class PrerequisiteRankerModel(nn.Module):
                def __init__(self, encoder_name, use_regression=True):
                    super(PrerequisiteRankerModel, self).__init__()
                    self.encoder = AutoModel.from_pretrained(encoder_name)
                    self.use_regression = use_regression
                    hidden_size = self.encoder.config.hidden_size

                    if use_regression:
                        # self.regressor = nn.Sequential(
                        #     nn.Linear(hidden_size, 256),
                        #     nn.ReLU(),
                        #     nn.Dropout(0.1),
                        #     nn.Linear(256, 1)
                        # )
                        # self.regressor = nn.Sequential(
                        #     nn.Linear(hidden_size, 384),  # must be 384
                        #     nn.ReLU(),
                        #     nn.Dropout(0.1),
                        #     nn.Linear(384, 1)
                        # )
                        self.regressor = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size // 2),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_size // 2, 1)
                        )
                    else:
                        self.classifier = nn.Sequential(
                            nn.Dropout(0.1),
                            nn.Linear(hidden_size, 256),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(256, 4)
                        )

                def forward(self, input_ids, attention_mask, token_type_ids=None):
                    outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids if token_type_ids is not None else None
                    )
                    cls_output = outputs.last_hidden_state[:, 0, :]

                    if self.use_regression:
                        output = self.regressor(cls_output)
                        return output.squeeze(-1)
                    else:
                        logits = self.classifier(cls_output)
                        return logits

            # Create model and load weights
            self.model = PrerequisiteRankerModel(encoder_name=self.encoder_name, use_regression=self.use_regression)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"Encoder model loaded from {model_path} (regression={self.use_regression})")

        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

       
    def rank_expressions(self, 
                         filtered_expressions: Dict[str, float], 
                         document_text: str,
                         batch_size: int = 8) -> Dict[str, int]:
        """
        Rank expressions by importance using the encoder model
        
        Args:
            filtered_expressions: Dictionary of expressions and their similarity scores
            document_text: Full text of the document
            batch_size: Number of expressions to process at once
            
        Returns:
            Dictionary of expressions mapped to their importance scores (0-3)
        """
        if not filtered_expressions:
            return {}
            
        # Prepare data
        expressions = list(filtered_expressions.keys())
        ranks = {}
        
        # Process in batches to avoid memory issues
        for i in range(0, len(expressions), batch_size):
            batch_expressions = expressions[i:i+batch_size]
            
            # Prepare inputs
            batch_inputs = []
            for expr in batch_expressions:
                # Truncate document if needed to avoid tokenizer limits
                if len(document_text) > 5000:
                    begin_text = document_text[:2000]
                    end_text = document_text[-2000:]
                    doc_text = begin_text + "..." + end_text
                else:
                    doc_text = document_text
                    
                # Encode
                encoding = self.tokenizer(
                    expr,
                    doc_text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                batch_inputs.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze()
                })
                
            # Stack inputs
            input_ids = torch.stack([item['input_ids'] for item in batch_inputs]).to(self.device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch_inputs]).to(self.device)
            
            if 'token_type_ids' in batch_inputs[0]:
                token_type_ids = torch.stack([item['token_type_ids'] for item in batch_inputs]).to(self.device)
            else:
                token_type_ids = None
                
            # Predict
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                if self.use_regression:
                    # For regression: round to nearest integer and clamp to valid range
                    preds = torch.round(outputs).clamp(0, 3).int()
                else:
                    # For classification: take argmax
                    _, preds = torch.max(outputs, dim=1)
                
            # Store predictions
            for j, expr in enumerate(batch_expressions):
                ranks[expr] = preds[j].item()
                
        # Sort by rank (descending) then by similarity score (descending)
        sorted_ranks = dict(sorted(
            ranks.items(),
            key=lambda item: (item[1], filtered_expressions[item[0]]),
            reverse=True
        ))
        
        return sorted_ranks

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
                         ranker: PrerequisiteRankerEncoder) -> Dict[str, int]:
    """
    Process a single page: load raw text and Stage B results, rank expressions, save Stage C results.
    
    Args:
        page_title: Title of the Wikipedia page
        raw_data_dir: Directory containing raw page text
        stage_b_output_dir: Directory containing filtered expressions from Stage B
        stage_c_output_dir: Directory to save ranked prerequisites
        ranker: Initialized encoder ranker
        
    Returns:
        Dictionary of ranked expressions
    """
    # Load raw document text
    raw_file_path = os.path.join(raw_data_dir, f"{page_title}.txt")
    if not os.path.exists(raw_file_path):
        print(f"Warning: Raw text file not found for {page_title}: {raw_file_path}")
        return {}
        
    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
    except Exception as e:
        print(f"Error reading raw text for {page_title}: {e}")
        return {}
        
    # Load Stage B filtered expressions
    stage_b_file_path = os.path.join(stage_b_output_dir, f"{page_title}_filtered.json")
    if not os.path.exists(stage_b_file_path):
        print(f"Warning: Stage B file not found for {page_title}: {stage_b_file_path}")
        return {}
        
    try:
        with open(stage_b_file_path, 'r', encoding='utf-8') as f:
            filtered_expressions = json.load(f)
    except Exception as e:
        print(f"Error loading Stage B data for {page_title}: {e}")
        return {}
        
    if not filtered_expressions:
        print(f"No filtered expressions found for {page_title}")
        return {}
        
    print(f"Processing {page_title}: {len(filtered_expressions)} expressions")
    
    # Rank expressions
    ranked_expressions = ranker.rank_expressions(filtered_expressions, document_text)
    
    # Save results
    output_path = os.path.join(stage_c_output_dir, f"{page_title}_ranked.json")
    save_ranked_expressions(ranked_expressions, output_path)
    
    return ranked_expressions

def main():
    """
    Main function for ranking expressions using the encoder-based model
    """
    # Configuration
    model_path = "models/stage_c_ranker_encoder_penalty.pt"
    raw_data_dir = "data/raw/raw_texts"
    stage_b_output_dir = "data/processed/stage_b"
    stage_c_output_dir = "data/processed/stage_c_encoder"
    
    print("=== Stage C: Encoder-based Expression Ranking ===")
    print(f"Model path: {model_path}")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Stage B output directory: {stage_b_output_dir}")
    print(f"Stage C output directory: {stage_c_output_dir}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the encoder model first using train_ranker_encoder.py")
        return
        
    # Initialize ranker
    try:
        ranker = PrerequisiteRankerEncoder(model_path)
    except Exception as e:
        print(f"Error initializing ranker: {e}")
        return
        
    # Create output directory
    os.makedirs(stage_c_output_dir, exist_ok=True)
    
    # Process all pages that have Stage B results
    if not os.path.exists(stage_b_output_dir):
        print(f"Error: Stage B output directory not found: {stage_b_output_dir}")
        return
        
    stage_b_files = [f for f in os.listdir(stage_b_output_dir) if f.endswith('_filtered.json')]
    
    if not stage_b_files:
        print("No Stage B files found to process")
        return
        
    print(f"Found {len(stage_b_files)} pages to process")
    
    # Process each page
    all_results = {}
    for stage_b_file in tqdm.tqdm(stage_b_files, desc="Processing pages"):
        page_title = stage_b_file.replace('_filtered.json', '')
        
        try:
            ranked_expressions = process_page_ranking(
                page_title=page_title,
                raw_data_dir=raw_data_dir,
                stage_b_output_dir=stage_b_output_dir,
                stage_c_output_dir=stage_c_output_dir,
                ranker=ranker
            )
            
            if ranked_expressions:
                all_results[page_title] = ranked_expressions
                
        except Exception as e:
            print(f"Error processing {page_title}: {e}")
            continue
            
    print(f"\nProcessing complete. Successfully processed {len(all_results)} pages.")
    print(f"Results saved to {stage_c_output_dir}")

if __name__ == "__main__":
    main()
