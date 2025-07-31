"""
Improved content domain filter with contextual heuristics
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from typing import List, Dict, Tuple
sys.path.append('src/util')
from src.util.get_raw_text import get_raw_text

class ContentDomainFilter:
    """
    Filters expressions based on their relevance to the content domain of the text.
    Enhanced with additional heuristics for detecting side mentions and contextual importance.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 similarity_threshold: float = 0.5, 
                 bert_model=None, tokenizer=None):
        """
        Initialize the filter with a language model and enhanced heuristics.
        
        Args:
            model_name: Name of the model to download from HuggingFace
            similarity_threshold: Minimum similarity threshold for accepting an expression
            bert_model: Pre-loaded BERT model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        
        if bert_model is not None and tokenizer is not None:
            self.model = bert_model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Patterns for detecting side mentions - focus on syntactic patterns only
        self.side_mention_patterns = [
            r'sometimes.*?{phrase}',
            r'often.*?{phrase}', 
            r'occasionally.*?{phrase}',
            r'for example.*?{phrase}',
            r'such as.*?{phrase}',
            r'including.*?{phrase}',
            r'also.*?{phrase}',
            r'meanwhile.*?{phrase}',
            r'in addition.*?{phrase}',
            r'as well.*?{phrase}',
            r'would.*?{phrase}.*?(during|while)',
            r'even.*?{phrase}',
            r'perhaps.*?{phrase}',
            r'might.*?{phrase}',
            r'\b(he|she|they|people|someone|researchers?|scientists?|students?)\s+(also\s+)?(enjoyed?|liked?|preferred?|loved?).*?{phrase}',
        ]
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text using the language model.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings[0]
    
    def _calculate_position_bonus(self, phrase: str, document: str) -> float:
        """Calculate bonus based on phrase position in document"""
        bonus = 0.0
        phrase_lower = phrase.lower()
        doc_lower = document.lower()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in document.split('\n') if p.strip()]
        if not paragraphs:
            return 0.0
            
        # First 100 words of document
        first_100_words = ' '.join(document.split()[:100]).lower()
        if phrase_lower in first_100_words:
            bonus += 0.2
            
        # First paragraph
        if phrase_lower in paragraphs[0].lower():
            bonus += 0.15
            
        # Title/first sentence
        first_sentence = document.split('.')[0].lower()
        if phrase_lower in first_sentence:
            bonus += 0.3
            
        return min(bonus, 0.4)  # Cap maximum bonus
    
    def _calculate_frequency_bonus(self, phrase: str, document: str) -> float:
        """Calculate bonus based on phrase frequency"""
        phrase_lower = phrase.lower()
        doc_lower = document.lower()
        
        # Count occurrences
        count = doc_lower.count(phrase_lower)
        
        if count > 3:
            return 0.2
        elif count > 1:
            return 0.1
        else:
            return 0.0
    
    def _detect_side_mention_penalty(self, phrase: str, document: str) -> float:
        """Detect side mentions using only syntactic patterns, not content words"""
        penalty = 0.0
        phrase_lower = phrase.lower()
        doc_lower = document.lower()
        
        # Search for syntactic side mention patterns
        for pattern in self.side_mention_patterns:
            # Replace {phrase} with the specific phrase
            specific_pattern = pattern.replace('{phrase}', re.escape(phrase_lower))
            if re.search(specific_pattern, doc_lower, re.IGNORECASE):
                penalty += 0.1
        
        # Check for parenthetical mentions (often side notes)
        parenthetical_pattern = r'\([^)]*' + re.escape(phrase_lower) + r'[^)]*\)'
        if re.search(parenthetical_pattern, doc_lower):
            penalty += 0.1
            
        # Check if phrase appears only in subordinate clauses
        # Look for patterns like "..., who also liked X, ..." or "..., which included X, ..."
        subordinate_patterns = [
            r',\s*who.*?' + re.escape(phrase_lower) + r'.*?,',
            r',\s*which.*?' + re.escape(phrase_lower) + r'.*?,',
            r',\s*where.*?' + re.escape(phrase_lower) + r'.*?,',
        ]
        
        for pattern in subordinate_patterns:
            if re.search(pattern, doc_lower):
                penalty += 0.05
                
        return min(penalty, 0.25)
    
    def _calculate_title_similarity_bonus(self, phrase: str, document: str) -> float:
        """Bonus if phrase is similar to title/main topic"""
        # Assume first sentence is title or abstract
        first_sentence = document.split('.')[0]
        
        phrase_embedding = self._get_embedding(phrase)
        title_embedding = self._get_embedding(first_sentence)
        
        similarity = cosine_similarity(
            phrase_embedding.reshape(1, -1),
            title_embedding.reshape(1, -1)
        )[0][0]
        
        if similarity > 0.6:
            return 0.2
        elif similarity > 0.4:
            return 0.1
        else:
            return 0.0
    
    def _calculate_context_richness_bonus(self, phrase: str, document: str) -> float:
        """Bonus for phrases appearing in rich, detailed contexts"""
        phrase_lower = phrase.lower()
        doc_lower = document.lower()
        
        # Find sentences containing the phrase
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        phrase_sentences = [s for s in sentences if phrase_lower in s.lower()]
        
        if not phrase_sentences:
            return 0.0
            
        # Calculate average length of sentences containing the phrase
        avg_length = np.mean([len(s.split()) for s in phrase_sentences])
        
        # Bonus for longer sentences (more detailed)
        if avg_length > 20:
            return 0.1
        elif avg_length > 15:
            return 0.05
        else:
            return 0.0
    
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
            doc_emb_reshaped = document_embedding.reshape(1, -1)
            expr_emb_reshaped = expr_emb.reshape(1, -1)
            similarity = cosine_similarity(doc_emb_reshaped, expr_emb_reshaped)[0][0]
            similarities[expr] = float(similarity)
        return similarities
    
    def _calculate_final_score(self, phrase: str, document: str) -> float:
        """Calculate final score using all heuristics"""
        # Base BERT similarity score
        doc_embedding = self._get_embedding(document)
        phrase_embedding = self._get_embedding(phrase)
        base_score = cosine_similarity(
            doc_embedding.reshape(1, -1),
            phrase_embedding.reshape(1, -1)
        )[0][0]
        
        # Additional heuristics
        position_bonus = self._calculate_position_bonus(phrase, document)
        frequency_bonus = self._calculate_frequency_bonus(phrase, document) 
        side_mention_penalty = self._detect_side_mention_penalty(phrase, document)
        title_similarity_bonus = self._calculate_title_similarity_bonus(phrase, document)
        context_richness_bonus = self._calculate_context_richness_bonus(phrase, document)
        
        # Calculate final score
        final_score = (base_score + 
                      position_bonus + 
                      frequency_bonus + 
                      title_similarity_bonus +
                      context_richness_bonus -
                      side_mention_penalty)
        
        return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
    
    def filter_expressions(self, document_text: str, expressions: List[str]) -> Dict[str, float]:
        """
        Filter expressions based on context and position - enhanced version.
        
        Args:
            document_text: Full text of the document
            expressions: List of expressions (potential links) to filter
            
        Returns:
            Dictionary of expressions that pass the similarity threshold along with their scores
        """
        if not expressions:
            return {}
            
        # Calculate scores for all expressions using enhanced heuristics
        scores = {}
        
        for expr in expressions:
            final_score = self._calculate_final_score(expr, document_text)
            scores[expr] = final_score
        
        # Calculate adaptive threshold
        score_values = list(scores.values())
        if score_values:
            adaptive_threshold = max(
                self.similarity_threshold,
                np.percentile(score_values, 40)  # Top 60% of scores
            )
        else:
            adaptive_threshold = self.similarity_threshold
            
        # Filter based on adaptive threshold
        filtered_expressions = {
            expr: score for expr, score in scores.items() 
            if score >= adaptive_threshold
        }
        
        # Sort by score (descending)
        filtered_expressions = dict(sorted(
            filtered_expressions.items(), 
            key=lambda item: item[1], 
            reverse=True
        ))
        
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

def process_page(page_title: str,
                 raw_data_dir: str,
                 stage_a_output_dir: str,
                 stage_b_output_dir: str,
                 filter_model: ContentDomainFilter) -> Dict[str, float]:
    """
    Process a single page through the filter.
    """
    document_text = get_raw_text(page_title)

    # Load expressions identified by Stage A
    csv_path = os.path.join(stage_a_output_dir, f"{page_title}.csv")
    df = pd.read_csv(csv_path)
    expressions = df['phrase'].dropna().tolist()

    # Filter expressions
    filtered_expressions = filter_model.filter_expressions(document_text, expressions)

    # Identify and print omitted expressions
    omitted_expressions = [expr for expr in expressions if expr not in filtered_expressions]
    if omitted_expressions:
        print(f"\nOmitted expressions for page '{page_title}':")
        for expr in omitted_expressions:
            print(f"  - {expr}")

    # Ensure output directory exists
    os.makedirs(stage_b_output_dir, exist_ok=True)

    # Save results
    output_path = os.path.join(stage_b_output_dir, f"{page_title}_filtered.json")
    filter_model.save_results(filtered_expressions, output_path)

    return filtered_expressions


def main():
    raw_data_dir = "data/raw"
    stage_a_output_dir = "data/processed/stage_a"
    stage_b_output_dir = "data/processed/stage_b"
    model_name = "bert-base-uncased"
    similarity_threshold = 0.5

    filter_model = ContentDomainFilter(model_name=model_name,
                                       similarity_threshold=similarity_threshold)

    page_titles = [f[:-4] for f in os.listdir(stage_a_output_dir) if f.endswith('.csv')]

    for page_title in page_titles:
        print(f"Processing page: {page_title}")
        filtered_expressions = process_page(page_title,
                                            raw_data_dir,
                                            stage_a_output_dir,
                                            stage_b_output_dir,
                                            filter_model)

        print(f"Filtered {len(filtered_expressions)} expressions for page {page_title}")


if __name__ == "__main__":
    main()
