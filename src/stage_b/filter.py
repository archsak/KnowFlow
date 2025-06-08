import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple, Optional

class ContentDomainFilter:
    """
    Filters expressions based on their relevance to the content domain of the text.
    Takes a list of expressions (identified potential links) and filters only those
    that are sufficiently related to the text subject through cosine similarity of embeddings.
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", similarity_threshold: float = 0.5):
        """
        Initialize the filter with a language model supporting the target language.
        
        Args:
            model_name: Name of the model to download from HuggingFace
            similarity_threshold: Minimum cosine similarity threshold for accepting an expression
                                  as belonging to the content domain
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text using the language model.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use mean of last hidden state as embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings[0]
    
    def get_document_embedding(self, document_text: str) -> np.ndarray:
        """
        Generate embedding for the entire document.
        
        Args:
            document_text: Full text of the document
            
        Returns:
            Document embedding vector
        """
        return self._get_embedding(document_text)
    
    def get_expression_embeddings(self, expressions: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all expressions.
        
        Args:
            expressions: List of expressions to generate embeddings for
            
        Returns:
            Dictionary mapping expressions to their embedding vectors
        """
        expression_embeddings = {}
        for expr in expressions:
            expression_embeddings[expr] = self._get_embedding(expr)
        return expression_embeddings
    
    def compute_similarities(self, document_embedding: np.ndarray, 
                            expression_embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute cosine similarity between document embedding and each expression embedding.
        
        Args:
            document_embedding: Document embedding vector
            expression_embeddings: Dictionary of expression embeddings
            
        Returns:
            Dictionary mapping expressions to their similarity scores
        """
        similarities = {}
        for expr, expr_emb in expression_embeddings.items():
            # Reshape for cosine_similarity function
            doc_emb_reshaped = document_embedding.reshape(1, -1)
            expr_emb_reshaped = expr_emb.reshape(1, -1)
            
            # Compute cosine similarity
            similarity = cosine_similarity(doc_emb_reshaped, expr_emb_reshaped)[0][0]
            similarities[expr] = float(similarity)
        
        return similarities
    
    def filter_expressions(self, document_text: str, expressions: List[str]) -> Dict[str, float]:
        """
        Filter expressions based on their similarity to the document content.
        
        Args:
            document_text: Full text of the document
            expressions: List of expressions (potential links) to filter
            
        Returns:
            Dictionary of expressions that pass the similarity threshold along with their scores
        """
        # Get document embedding
        document_embedding = self.get_document_embedding(document_text)
        
        # Get embeddings for all expressions
        expression_embeddings = self.get_expression_embeddings(expressions)
        
        # Compute similarities
        similarities = self.compute_similarities(document_embedding, expression_embeddings)
        
        # Filter by threshold
        filtered_expressions = {expr: score for expr, score in similarities.items() 
                               if score >= self.similarity_threshold}
        
        # Sort by similarity score (descending)
        filtered_expressions = dict(sorted(filtered_expressions.items(), 
                                          key=lambda item: item[1], 
                                          reverse=True))
        
        return filtered_expressions
    
    def save_results(self, filtered_expressions: Dict[str, float], output_path: str):
        """
        Save filtered expressions to a JSON file.
        
        Args:
            filtered_expressions: Dictionary of filtered expressions with their similarity scores
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_expressions, f, ensure_ascii=False, indent=4)
        
        print(f"Results saved to {output_path}")


def process_wiki_page(page_id: str, 
                     raw_data_dir: str, 
                     stage_a_output_dir: str,
                     stage_b_output_dir: str,
                     filter_model: ContentDomainFilter) -> Dict[str, float]:
    """
    Process a single Wikipedia page through the filter.
    
    Args:
        page_id: ID of the Wikipedia page
        raw_data_dir: Directory containing raw Wikipedia pages
        stage_a_output_dir: Directory containing Stage A outputs (identified expressions)
        stage_b_output_dir: Directory to save Stage B outputs
        filter_model: Initialized ContentDomainFilter instance
        
    Returns:
        Dictionary of filtered expressions with their similarity scores
    """
    # Load the raw document text
    with open(os.path.join(raw_data_dir, f"{page_id}.txt"), 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Load expressions identified by Stage A
    with open(os.path.join(stage_a_output_dir, f"{page_id}_links.json"), 'r', encoding='utf-8') as f:
        expressions = json.load(f)
    
    # Filter expressions
    filtered_expressions = filter_model.filter_expressions(document_text, expressions)
    
    # Ensure output directory exists
    os.makedirs(stage_b_output_dir, exist_ok=True)
    
    # Save results
    output_path = os.path.join(stage_b_output_dir, f"{page_id}_filtered.json")
    filter_model.save_results(filtered_expressions, output_path)
    
    return filtered_expressions


def main():
    """
    Main function to process all Wikipedia pages.
    """
    # Configuration
    raw_data_dir = "data/raw"
    stage_a_output_dir = "data/processed/stage_a"
    stage_b_output_dir = "data/processed/stage_b"
    model_name = "bert-base-multilingual-cased"  # Use appropriate model for your language
    similarity_threshold = 0.5  # Adjust based on experimentation
    
    # Initialize the content domain filter
    filter_model = ContentDomainFilter(model_name=model_name, 
                                      similarity_threshold=similarity_threshold)
    
    # Get all page IDs from Stage A output
    page_ids = [f.split('_')[0] for f in os.listdir(stage_a_output_dir) 
               if f.endswith('_links.json')]
    
    # Process each page
    for page_id in page_ids:
        print(f"Processing page: {page_id}")
        filtered_expressions = process_wiki_page(page_id, 
                                               raw_data_dir, 
                                               stage_a_output_dir, 
                                               stage_b_output_dir, 
                                               filter_model)
        
        print(f"Filtered {len(filtered_expressions)} expressions for page {page_id}")


if __name__ == "__main__":
    main()
