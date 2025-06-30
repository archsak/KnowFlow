"""
KnowFlow: BERT-based Link Phrase Detection
Main implementation file for the KnowFlow project

Dependencies:
pip install transformers torch datasets scikit-learn numpy
"""


print("✅ Sanity check: Bert.py was imported and ran")


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import json
import re
from typing import List, Dict, Tuple, Optional
import argparse
import os
import time

from WikipediaExtractor import WikipediaExtractor, get_diverse_article_titles, get_random_articles
from EvaluationResult import GoldStandardEvaluator
from typing import List, Dict, Set
from collections import Counter

class LinkDetectionDataset(Dataset):
    """PyTorch Dataset for BERT-based link detection."""
    
    def __init__(self, phrases: List[str], contexts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 128):
        self.phrases = phrases
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self, idx):
        phrase = str(self.phrases[idx])
        context = str(self.contexts[idx])
        label = self.labels[idx]
        
        # Encode: [CLS] context [SEP] phrase [SEP]
        encoding = self.tokenizer(
            context,
            phrase,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTLinkClassifier(nn.Module):
    """BERT-based classifier for link phrase detection."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


class TrainingDataProcessor:
    """Processes Wikipedia articles into training data for link detection."""
    
    def __init__(self, extractor: WikipediaExtractor):
        self.extractor = extractor
        # Define stopwords and common non-linkable phrases
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'very', 'much', 'many', 'most', 'more',
            'some', 'any', 'all', 'each', 'every', 'no', 'not', 'only', 'just'
        }
    def create_training_data(self, articles: List[Dict], 
                           negative_ratio: float = 1.0,  # Reduced from 2.0
                           max_phrase_length: int = 4,   # Reduced from 5
                           min_phrase_length: int = 1) -> tuple:
        """Enhanced training data creation with better filtering."""
        
        phrases = []
        contexts = []
        labels = []
        
        # Collect all positive phrases first to understand patterns
        all_positive_phrases = set()
        for article in articles:
            for link in article['links']:
                phrase = link['display_text'].strip()
                if self._is_valid_phrase(phrase, min_phrase_length, max_phrase_length):
                    all_positive_phrases.add(phrase.lower())
        
        print(f"Total unique positive phrases: {len(all_positive_phrases)}")
        
        for i, article in enumerate(articles):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(articles)}: {article['title']}")
            
            clean_text = article['clean_text']
            links = article['links']
            
            # Process positive examples with stricter filtering
            positive_phrases = []
            for link in links:
                phrase = link['display_text'].strip()
                
                if self._is_valid_phrase(phrase, min_phrase_length, max_phrase_length):
                    context = self._get_context_around_phrase(clean_text, phrase)
                    if context:
                        phrases.append(phrase)
                        contexts.append(context)
                        labels.append(1)
                        positive_phrases.append(phrase.lower())
            
            # Generate higher-quality negative examples
            negative_candidates = self._extract_smart_negative_candidates(
                clean_text, positive_phrases, all_positive_phrases, 
                max_phrase_length, min_phrase_length
            )
            
            # Sample negatives with better selection
            num_negatives = min(
                int(len(positive_phrases) * negative_ratio),
                len(negative_candidates)
            )
            
            if num_negatives > 0:
                # Use weighted sampling to prefer better negative examples
                selected_negatives = self._select_best_negatives(
                    negative_candidates, num_negatives
                )
                
                for phrase in selected_negatives:
                    context = self._get_context_around_phrase(clean_text, phrase)
                    if context:
                        phrases.append(phrase)
                        contexts.append(context)
                        labels.append(0)
        
        print(f"Enhanced training data created:")
        print(f"  Total samples: {len(phrases)}")
        print(f"  Positive samples: {sum(labels)}")
        print(f"  Negative samples: {len(labels) - sum(labels)}")
        print(f"  Positive/Negative ratio: 1:{negative_ratio}")
        
        return phrases, contexts, labels
    
    def _is_valid_phrase(self, phrase: str, min_length: int, max_length: int) -> bool:
        """Enhanced phrase validation."""
        if not phrase or len(phrase.strip()) < 3:
            return False
            
        words = phrase.split()
        word_count = len(words)
        
        # Length constraints
        if word_count < min_length or word_count > max_length:
            return False
        
        # Skip if all words are stopwords
        if all(word.lower() in self.stopwords for word in words):
            return False
        
        # Skip purely numeric or date patterns
        if re.match(r'^\d+$', phrase) or re.match(r'^\d{4}s?$', phrase):
            return False
        
        # Skip very common function words
        if phrase.lower() in {'the', 'this', 'that', 'these', 'those', 'such', 'other'}:
            return False
            
        return True
    
    def _extract_smart_negative_candidates(self, text: str, positive_phrases: List[str], 
                                         all_positive_phrases: Set[str],
                                         max_phrase_length: int, 
                                         min_phrase_length: int) -> List[str]:
        """Extract smarter negative candidates that are less likely to be actual links."""
        
        linked_phrases = set(phrase.lower() for phrase in positive_phrases)
        candidates = []
        words = text.split()
        
        for i in range(len(words)):
            for length in range(min_phrase_length, min(max_phrase_length + 1, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+length])
                phrase_clean = phrase.strip('.,!?;:()"\'')
                phrase_lower = phrase_clean.lower()
                
                # Skip if it's a known link or very similar to known links
                if (phrase_lower in linked_phrases or 
                    phrase_lower in all_positive_phrases or
                    self._too_similar_to_positives(phrase_lower, all_positive_phrases)):
                    continue
                
                # Apply enhanced filtering
                if self._is_good_negative_candidate(phrase_clean):
                    candidates.append(phrase_clean)
        
        return list(set(candidates))  # Remove duplicates
    
    def _too_similar_to_positives(self, phrase: str, positive_phrases: Set[str]) -> bool:
        """Check if phrase is too similar to known positive phrases."""
        words = set(phrase.split())
        
        # Check for exact substring matches
        for pos_phrase in positive_phrases:
            if phrase in pos_phrase or pos_phrase in phrase:
                return True
            
            # Check for high word overlap
            pos_words = set(pos_phrase.split())
            if len(words) > 1 and len(pos_words) > 1:
                overlap = len(words & pos_words) / min(len(words), len(pos_words))
                if overlap > 0.7:  # 70% word overlap threshold
                    return True
        
        return False
    
    def _is_good_negative_candidate(self, phrase: str) -> bool:
        """Enhanced filtering for negative candidates."""
        if not self._is_valid_phrase(phrase, 2, 4):
            return False
        
        # Skip common patterns that are rarely linked
        bad_patterns = [
            r'^(in|on|at|by|for|with|the|a|an)\s',  # Starting with prepositions/articles
            r'\s(and|or|but|so|yet|for|nor)$',      # Ending with conjunctions
            r'^\d+\s(years?|days?|months?|hours?)',  # Time expressions
            r'^(very|quite|rather|somewhat|extremely)', # Adverb phrases
            r'(said|says|stated|reported|announced)', # Reporting verbs
            r'^(according to|due to|because of)',     # Common phrases
        ]
        
        phrase_lower = phrase.lower()
        for pattern in bad_patterns:
            if re.search(pattern, phrase_lower):
                return False
        
        # Prefer phrases with at least one capitalized word (proper nouns)
        if not any(word[0].isupper() for word in phrase.split() if word):
            return False
            
        # Skip if phrase is too generic
        generic_words = {
            'method', 'system', 'process', 'technique', 'approach', 'way',
            'thing', 'part', 'piece', 'type', 'kind', 'form', 'example',
            'case', 'instance', 'situation', 'condition', 'state', 'level'
        }
        
        words = phrase_lower.split()
        if len(words) == 1 and words[0] in generic_words:
            return False
        
        return True
    
    def _select_best_negatives(self, candidates: List[str], num_needed: int) -> List[str]:
        """Select the best negative examples using scoring."""
        if len(candidates) <= num_needed:
            return candidates
        
        # Score candidates (higher score = better negative example)
        scored_candidates = []
        
        for candidate in candidates:
            score = self._score_negative_candidate(candidate)
            scored_candidates.append((candidate, score))
        
        # Sort by score and take top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in scored_candidates[:num_needed]]
    
    def _score_negative_candidate(self, phrase: str) -> float:
        """Score a negative candidate (higher = better negative example)."""
        score = 0.0
        words = phrase.split()
        
        # Prefer proper nouns (capitalized)
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        score += capitalized_words * 0.3
        
        # Prefer medium length phrases
        if 2 <= len(words) <= 3:
            score += 0.2
        
        # Bonus for technical-sounding terms
        technical_suffixes = ['-tion', '-sion', '-ment', '-ness', '-ity', '-ism', '-ology']
        if any(phrase.lower().endswith(suffix) for suffix in technical_suffixes):
            score += 0.15
        
        # Penalty for common words
        common_penalty = sum(0.1 for word in words if word.lower() in self.stopwords)
        score -= common_penalty
        
        return score
    
    def _get_context_around_phrase(self, text: str, phrase: str, window: int = 50) -> str:
        """Get context around a phrase - same as original but could be enhanced."""
        words = text.split()
        phrase_words = phrase.split()
        
        for i in range(len(words) - len(phrase_words) + 1):
            if ' '.join(words[i:i+len(phrase_words)]).lower() == phrase.lower():
                start = max(0, i - window)
                end = min(len(words), i + len(phrase_words) + window)
                return ' '.join(words[start:end])
        
        return ' '.join(words[:100])


class KnowFlowBERTDetector:
    """Main BERT-based link phrase detector for KnowFlow project."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = WikipediaExtractor()
        self.evaluator = GoldStandardEvaluator()
        self.data_processor = TrainingDataProcessor(self.extractor)
        
        print(f"KnowFlow BERT Detector initialized")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_name}")
    
    def get_wikipedia_data(self, num_articles: int = 50, use_cached: bool = True) -> List[Dict]:
        """
        Get Wikipedia data using WikipediaExtractor.
        
        Args:
            num_articles: Number of articles to fetch
            use_cached: Whether to use cached data if available
            
        Returns:
            List of articles with clean text and links
        """
        cache_filename = 'knowflow_training_data.json'
        
        # Try to load cached data first
        # if use_cached:
        #     try:
        #         cached_data = self.extractor.load_dataset(cache_filename)
        #         if cached_data and len(cached_data) >= num_articles:
        #             print(f"Using cached data: {len(cached_data)} articles")
        #             return cached_data[:num_articles]
        #     except:
        #         pass
        
        print("Fetching fresh Wikipedia data...")
        
        # Get diverse article titles
        article_titles = get_random_articles(num_articles)

        
        articles = self.extractor.get_training_dataset(
            article_titles, 
            save_cache=False  # We'll save with our own filename
        )
        
        if not articles:
            raise ValueError("Failed to fetch Wikipedia data!")
        
        # Save with our cache filename
        self.extractor.save_dataset(articles, cache_filename)
        
        return articles
    
    
    def train(self, articles: List[Dict] = None, epochs: int = 3, batch_size: int = 32, 
              learning_rate: float = 2e-5):
        """
        Train the BERT-based link detector.
        
        Args:
            articles: List of Wikipedia articles (if None, will fetch automatically)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        
        if articles is None:
            print("No articles provided, fetching Wikipedia data...")
            articles = self.get_wikipedia_data(num_articles=100)
        
        # Prepare training data using WikipediaExtractor results
        phrases, contexts, labels = self.data_processor.create_training_data(articles)
        
        if len(phrases) == 0:
            raise ValueError("No training data generated!")
        
        # Create dataset
        dataset = LinkDetectionDataset(phrases, contexts, labels, self.tokenizer)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = BERTLinkClassifier(self.model_name)
        self.model.to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Starting training:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Evaluate on test set
        print("\nEvaluating on test set:")
        self._evaluate_test_set(test_loader)
        
        print("Training completed!")
    
    def _evaluate_test_set(self, test_loader):
        """Evaluate the model on test set."""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(true_labels, predictions, 
                                  target_names=['Non-Link', 'Link']))
    
    def predict_links(self, text: str, threshold: float = 0.5) -> List[Dict]:
        """
        Predict links for a given text.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for predictions
            
        Returns:
            List of dictionaries with 'phrase' and 'confidence' keys
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")
        
        # Extract candidate phrases
        candidates = self._extract_candidates(text)
        
        results = []
        self.model.eval()
        
        with torch.no_grad():
            for phrase in candidates:
                context = self._get_context(text, phrase)
                
                # Tokenize
                encoding = self.tokenizer(
                    context,
                    phrase,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs, dim=1)
                link_prob = probabilities[0][1].item()  # Probability of being a link
                
                if link_prob >= threshold:
                    results.append({
                        'phrase': phrase,
                        'confidence': link_prob
                    })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results
    
    def evaluate_with_gold_standard(self, test_articles: List[str] = None, 
                                   confidence_threshold: float = 0.7) -> Dict:
        """
        Evaluate the model using gold standard Wikipedia articles.
        
        Args:
            test_articles: List of Wikipedia article titles to evaluate on
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")
        
        if test_articles is None:
            # Use a different set of articles for testing
            test_articles = [
                # "Prime Number",
                # "Attention Is All You Need", 
                "BERT (language model)", 
                # "GPT (language model)", 
                # "AlexNet", 
                # "Word2vec", 
                # "U-Net",
                # "Capsule neural network",
                # "Neural differential equation", 
                # "DeepDream", 
                # "Batch normalization", 
                # "Swish function", 
                # "Microhistory",
                # "Eurasia Group",
                # "Miranda v. Arizona",
                # "Cook–Levin theorem",   
                # "Beowulf: The Monsters and the Critics",
                # "Pathetic fallacy",
            ]
        
        print(f"Evaluating model with gold standard on {len(test_articles)} articles...")
        
        # Use the GoldStandardEvaluator
        results = self.evaluator.evaluate_model(
            self, test_articles, confidence_threshold
        )
        
        # Print detailed report
        self.evaluator.print_evaluation_report(results)
        
        return results
    
    # def _extract_candidates(self, text: str) -> List[str]:
    #     """Extract candidate phrases from text."""
    #     candidates = set()
    #     words = text.split()
        
    #     # Extract various n-grams as candidates
    #     for i in range(len(words)):
    #         # Single words (capitalized, longer than 2 chars)
    #         word = words[i].strip('.,!?;:()"\'')
    #         if len(word) > 2 :
    #             candidates.add(word)
            
    #         # Bigrams
    #         if i < len(words) - 1:
    #             bigram = f"{words[i]} {words[i+1]}"
    #             bigram_clean = bigram.strip('.,!?;:()"\'')
    #             if len(bigram_clean) > 4:
    #                 candidates.add(bigram_clean)
            
    #         # Trigrams
    #         if i < len(words) - 2:
    #             trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
    #             trigram_clean = trigram.strip('.,!?;:()"\'')
    #             if len(trigram_clean) > 6:
    #                 candidates.add(trigram_clean)
        
    #     print('candidates: ', candidates)
    #     return list(candidates)
    
    
    def _extract_candidates(self, text: str) -> List[str]:
        """Generate realistic candidate phrases based on training-time filtering."""
        words = text.split()
        processor = self.data_processor
        max_phrase_length = 4
        min_phrase_length = 1

        candidates = set()

        for i in range(len(words)):
            for length in range(min_phrase_length, max_phrase_length + 1):
                if i + length > len(words):
                    break
                phrase = ' '.join(words[i:i+length])
                phrase_clean = phrase.strip('.,!?;:()"\'')
                
                if processor._is_valid_phrase(phrase_clean, min_phrase_length, max_phrase_length):
                    candidates.add(phrase_clean)

        print("num candidates: ", len(candidates))
        return list(candidates)
    
    def _get_context(self, text: str, phrase: str, window: int = 50) -> str:
        """Get context around a phrase."""
        words = text.split()
        phrase_words = phrase.split()
        
        # Find phrase in text (case insensitive)
        for i in range(len(words) - len(phrase_words) + 1):
            if ' '.join(words[i:i+len(phrase_words)]).lower() == phrase.lower():
                start = max(0, i - window)
                end = min(len(words), i + len(phrase_words) + window)
                return ' '.join(words[start:end])
        
        # Fallback to beginning of text
        return ' '.join(words[:100])
    
    def save_model(self, filepath: str = 'knowflow_bert_model.pth'):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save!")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'knowflow_bert_model.pth'):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model = BERTLinkClassifier(checkpoint['model_name'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {filepath}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='KnowFlow: BERT-based Link Phrase Detection')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], 
                       default='train', help='Mode to run')
    parser.add_argument('--articles', type=int, default=50, 
                       help='Number of articles to fetch for training')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Training batch size')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Confidence threshold for predictions')
    parser.add_argument('--text_file', type=str, help='Path to .txt file to analyze (for predict mode)')

    parser.add_argument('--model_path', type=str, default='knowflow_bert_model.pth',
                       help='Path to save/load model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        detector = KnowFlowBERTDetector()
        articles = detector.get_wikipedia_data(num_articles=args.articles)
        detector.train(articles, epochs=args.epochs, batch_size=args.batch_size)
        detector.save_model(args.model_path)
        
    elif args.mode == 'predict':

        if not args.text_file:
            print("Please provide a path to a text file with --text_file")
            return

        if not os.path.exists(args.text_file):
            print(f"File not found: {args.text_file}")
            return

        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        detector = KnowFlowBERTDetector()
        detector.load_model(args.model_path)

        predicted_links = detector.predict_links(text, threshold=args.threshold)

        # print("Predicted links:")
        # for link in predicted_links:
        #     print(f"  • '{link['phrase']}' (confidence: {link['confidence']:.3f})")

        
    elif args.mode == 'evaluate':

        detector = KnowFlowBERTDetector()
        detector.load_model(args.model_path)
        detector.evaluate_with_gold_standard(confidence_threshold=args.threshold)



if __name__ == "__main__":
    print("hi")
    main()