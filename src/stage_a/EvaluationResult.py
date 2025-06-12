"""
Gold Standard Evaluation System for KnowFlow Project
Evaluates link phrase detection against actual Wikipedia links
Enhanced with file output and section filtering
"""

import numpy as np
import json
import csv
from typing import Dict, List, Set
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import os
from WikipediaExtractor import WikipediaExtractor


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    
    def __str__(self):
        return f"P: {self.precision:.3f}, R: {self.recall:.3f}, F1: {self.f1_score:.3f}"


@dataclass
class ArticleEvaluation:
    """Evaluation results for a single article."""
    title: str
    metrics: EvaluationMetrics
    predicted_phrases: List[str]
    actual_phrases: List[str]
    true_positives: List[str]
    false_positives: List[str]
    false_negatives: List[str]


class GoldStandardEvaluator:
    """
    Evaluates link detection models against real Wikipedia articles as gold standard.
    Enhanced with file output and section filtering.
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.extractor = WikipediaExtractor()
        self.output_dir = output_dir
        self.stop_headings = {
            "See_also", "Notes", "References", "Further_reading", "External_links",
            "Citations", "Bibliography", "Sources"
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_article(self, title: str, predicted_links: List[Dict]) -> ArticleEvaluation:
        """
        Evaluate predictions for a single article against Wikipedia gold standard.
        
        Args:
            title: Wikipedia article title
            predicted_links: List of dicts with 'phrase' and 'confidence' keys
            
        Returns:
            ArticleEvaluation object with detailed metrics
        """
        # Get gold standard data with section filtering
        article_data = self.extractor.get_article_data(title)
        if not article_data:
            raise ValueError(f"Could not fetch article: {title}")
        
        # Extract and normalize phrases
        predicted_phrases = set()
        for link in predicted_links:
            phrase = self._normalize_phrase(link.get('phrase', ''))
            if phrase:
                predicted_phrases.add(phrase)
        
        actual_phrases = set()
        for link in article_data['links']:
            phrase = self._normalize_phrase(link['display_text'])
            if phrase:
                actual_phrases.add(phrase)
        
        # Calculate metrics
        true_positives_set = predicted_phrases & actual_phrases
        false_positives_set = predicted_phrases - actual_phrases
        false_negatives_set = actual_phrases - predicted_phrases
        
        tp = len(true_positives_set)
        fp = len(false_positives_set)
        fn = len(false_negatives_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )
        
        return ArticleEvaluation(
            title=title,
            metrics=metrics,
            predicted_phrases=list(predicted_phrases),
            actual_phrases=list(actual_phrases),
            true_positives=list(true_positives_set),
            false_positives=list(false_positives_set),
            false_negatives=list(false_negatives_set)
        )
    
    def evaluate_model(self, model, test_articles: List[str], 
                      confidence_threshold: float = 0.7) -> Dict:
        """
        Comprehensive evaluation of a model on multiple articles.
        
        Args:
            model: Model with predict_links method
            test_articles: List of Wikipedia article titles
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary with evaluation results and statistics
        """
        article_evaluations = []
        successful_evaluations = 0
        
        print(f"Evaluating model on {len(test_articles)} articles...")
        
        # Create timestamp for this evaluation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"eval_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize comparison data for file output
        comparison_data = []
        
        for i, title in enumerate(test_articles):
            print(f"Processing {i+1}/{len(test_articles)}: {title}")
            
            try:
                # Get article data for model input (with section filtering)
                article_data = self.extractor.get_article_data(title)
                if not article_data:
                    print(f"    Skipping {title} - could not fetch")
                    continue
                
                # Get model predictions
                predicted_links = model.predict_links(
                    article_data['clean_text'], 
                    threshold=confidence_threshold
                )
                
                # Evaluate against gold standard
                evaluation = self.evaluate_article(title, predicted_links)
                article_evaluations.append(evaluation)
                successful_evaluations += 1
                
                # Collect data for file output
                self._collect_comparison_data(evaluation, comparison_data)
                
                print(f"   {evaluation.metrics}")
                
            except Exception as e:
                print(f"   Error processing {title}: {e}")
                continue
        
        if successful_evaluations == 0:
            return {"error": "No successful evaluations"}
        
        # Calculate aggregate statistics
        summary_stats = self._calculate_summary_statistics(article_evaluations)
        error_analysis = self._analyze_errors(article_evaluations)
        
        # Save detailed results to files
        self._save_evaluation_files(run_dir, article_evaluations, comparison_data, 
                                   summary_stats, error_analysis, confidence_threshold)
        
        results = {
            'summary': summary_stats,
            'article_evaluations': article_evaluations,
            'error_analysis': error_analysis,
            'evaluation_config': {
                'articles_evaluated': successful_evaluations,
                'confidence_threshold': confidence_threshold,
                'output_directory': run_dir
            }
        }
        
        return results
    
    def _collect_comparison_data(self, evaluation: ArticleEvaluation, comparison_data: List[Dict]):
        """Collect data for detailed comparison files."""
        
        # Add true positives
        for phrase in evaluation.true_positives:
            comparison_data.append({
                'article': evaluation.title,
                'phrase': phrase,
                'predicted': True,
                'actual': True,
                'result': 'TRUE_POSITIVE'
            })
        
        # Add false positives
        for phrase in evaluation.false_positives:
            comparison_data.append({
                'article': evaluation.title,
                'phrase': phrase,
                'predicted': True,
                'actual': False,
                'result': 'FALSE_POSITIVE'
            })
        
        # Add false negatives
        for phrase in evaluation.false_negatives:
            comparison_data.append({
                'article': evaluation.title,
                'phrase': phrase,
                'predicted': False,
                'actual': True,
                'result': 'FALSE_NEGATIVE'
            })
    
    def _save_evaluation_files(self, run_dir: str, article_evaluations: List[ArticleEvaluation],
                              comparison_data: List[Dict], summary_stats: Dict, 
                              error_analysis: Dict, confidence_threshold: float):
        """Save detailed evaluation results to multiple files."""
        
        # 1. Save detailed comparison CSV
        comparison_file = os.path.join(run_dir, "detailed_comparison.csv")
        with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
            if comparison_data:
                writer = csv.DictWriter(f, fieldnames=['article', 'phrase', 'predicted', 'actual', 'result'])
                writer.writeheader()
                writer.writerows(comparison_data)
        print(f"Detailed comparison saved to: {comparison_file}")
        
        # 2. Save article-by-article results
        article_results_file = os.path.join(run_dir, "article_results.csv")
        with open(article_results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Article', 'Precision', 'Recall', 'F1-Score', 'True Positives', 
                           'False Positives', 'False Negatives', 'Total Predicted', 'Total Actual'])
            
            for eval_result in article_evaluations:
                writer.writerow([
                    eval_result.title,
                    f"{eval_result.metrics.precision:.3f}",
                    f"{eval_result.metrics.recall:.3f}",
                    f"{eval_result.metrics.f1_score:.3f}",
                    eval_result.metrics.true_positives,
                    eval_result.metrics.false_positives,
                    eval_result.metrics.false_negatives,
                    len(eval_result.predicted_phrases),
                    len(eval_result.actual_phrases)
                ])
        print(f"Article results saved to: {article_results_file}")
        
        # 3. Save predicted vs actual phrases by article
        phrases_file = os.path.join(run_dir, "predicted_vs_actual_phrases.json")
        phrases_data = {}
        for eval_result in article_evaluations:
            phrases_data[eval_result.title] = {
                'predicted_phrases': sorted(eval_result.predicted_phrases),
                'actual_phrases': sorted(eval_result.actual_phrases),
                'true_positives': sorted(eval_result.true_positives),
                'false_positives': sorted(eval_result.false_positives),
                'false_negatives': sorted(eval_result.false_negatives)
            }
        
        with open(phrases_file, 'w', encoding='utf-8') as f:
            json.dump(phrases_data, f, indent=2, ensure_ascii=False)
        print(f"Phrases comparison saved to: {phrases_file}")
        
        # 4. Save summary report
        summary_file = os.path.join(run_dir, "evaluation_summary.json")
        full_summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'confidence_threshold': confidence_threshold,
            'summary_statistics': summary_stats,
            'error_analysis': error_analysis,
            'evaluation_config': {
                'stop_headings_filtered': list(self.stop_headings),
                'section_filtering_enabled': True
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(full_summary, f, indent=2, ensure_ascii=False)
        print(f"Summary report saved to: {summary_file}")
        
        # 5. Save error analysis details
        errors_file = os.path.join(run_dir, "error_analysis.csv")
        with open(errors_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # False positives
            writer.writerow(['FALSE POSITIVES'])
            writer.writerow(['Phrase', 'Frequency'])
            for phrase, count in error_analysis['most_common_false_positives']:
                writer.writerow([phrase, count])
            
            writer.writerow([])  # Empty row
            
            # False negatives
            writer.writerow(['FALSE NEGATIVES'])
            writer.writerow(['Phrase', 'Frequency'])
            for phrase, count in error_analysis['most_common_false_negatives']:
                writer.writerow([phrase, count])
        print(f"Error analysis saved to: {errors_file}")
    
    def _normalize_phrase(self, phrase: str) -> str:
        """Normalize phrase text for consistent comparison."""
        if not phrase:
            return ""
        
        # Convert to lowercase and strip
        normalized = phrase.lower().strip()
        
        # Remove punctuation from ends
        normalized = normalized.strip('.,!?;:()"\'')
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _calculate_summary_statistics(self, evaluations: List[ArticleEvaluation]) -> Dict:
        """Calculate aggregate statistics across all evaluations."""
        if not evaluations:
            return {}
        
        # Micro-averaged metrics (aggregate confusion matrix)
        total_tp = sum(e.metrics.true_positives for e in evaluations)
        total_fp = sum(e.metrics.false_positives for e in evaluations)
        total_fn = sum(e.metrics.false_negatives for e in evaluations)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        # Macro-averaged metrics (average of individual article metrics)
        macro_precision = np.mean([e.metrics.precision for e in evaluations])
        macro_recall = np.mean([e.metrics.recall for e in evaluations])
        macro_f1 = np.mean([e.metrics.f1_score for e in evaluations])
        
        return {
            'articles_evaluated': len(evaluations),
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total_predictions': sum(len(e.predicted_phrases) for e in evaluations),
            'total_gold_links': sum(len(e.actual_phrases) for e in evaluations),
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn
        }
    
    def _analyze_errors(self, evaluations: List[ArticleEvaluation]) -> Dict:
        """Analyze error patterns across evaluations."""
        all_false_positives = []
        all_false_negatives = []
        
        for evaluation in evaluations:
            all_false_positives.extend(evaluation.false_positives)
            all_false_negatives.extend(evaluation.false_negatives)
        
        # Pattern analysis
        fp_patterns = self._analyze_phrase_patterns(all_false_positives)
        fn_patterns = self._analyze_phrase_patterns(all_false_negatives)
        
        # Most common errors
        fp_counts = defaultdict(int)
        fn_counts = defaultdict(int)
        
        for phrase in all_false_positives:
            fp_counts[phrase] += 1
        
        for phrase in all_false_negatives:
            fn_counts[phrase] += 1
        
        return {
            'false_positive_patterns': fp_patterns,
            'false_negative_patterns': fn_patterns,
            'most_common_false_positives': sorted(fp_counts.items(), key=lambda x: x[1], reverse=True)[:20],
            'most_common_false_negatives': sorted(fn_counts.items(), key=lambda x: x[1], reverse=True)[:20],
            'total_false_positives': len(all_false_positives),
            'total_false_negatives': len(all_false_negatives)
        }
    
    def _analyze_phrase_patterns(self, phrases: List[str]) -> Dict:
        """Analyze patterns in a list of phrases."""
        patterns = {
            'single_word': 0,
            'two_words': 0,
            'multiple_words': 0,
            'capitalized': 0,
            'all_lowercase': 0,
            'contains_numbers': 0,
            'technical_terms': 0
        }
        
        technical_keywords = ['theory', 'principle', 'law', 'effect', 'method', 'algorithm', 'system']
        
        for phrase in phrases:
            word_count = len(phrase.split())
            
            if word_count == 1:
                patterns['single_word'] += 1
            elif word_count == 2:
                patterns['two_words'] += 1
            else:
                patterns['multiple_words'] += 1
            
            if phrase and phrase[0].isupper():
                patterns['capitalized'] += 1
            elif phrase.islower():
                patterns['all_lowercase'] += 1
            
            if any(char.isdigit() for char in phrase):
                patterns['contains_numbers'] += 1
            
            if any(keyword in phrase.lower() for keyword in technical_keywords):
                patterns['technical_terms'] += 1
        
        return patterns
    
    def print_evaluation_report(self, results: Dict):
        """Print a formatted evaluation report."""
        if 'error' in results:
            print(f"Evaluation failed: {results['error']}")
            return
        
        summary = results['summary']
        
        print("\n" + "="*60)
        print("KNOWFLOW EVALUATION REPORT")
        print("="*60)
        
        print(f"Articles Evaluated: {summary['articles_evaluated']}")
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Total Gold Links: {summary['total_gold_links']}")
        
        print(f"\nMICRO-AVERAGED METRICS (Overall Performance)")
        print(f"Precision: {summary['micro_precision']:.3f}")
        print(f"Recall: {summary['micro_recall']:.3f}")
        print(f"F1-Score: {summary['micro_f1']:.3f}")
        
        print(f"\nMACRO-AVERAGED METRICS (Average per Article)")
        print(f"Precision: {summary['macro_precision']:.3f}")
        print(f"Recall: {summary['macro_recall']:.3f}")
        print(f"F1-Score: {summary['macro_f1']:.3f}")
        
        # Show top performing articles
        print(f"\nTOP PERFORMING ARTICLES")
        print("-" * 40)
        top_articles = sorted(results['article_evaluations'], 
                            key=lambda x: x.metrics.f1_score, reverse=True)[:5]
        
        for article in top_articles:
            print(f"{article.title[:35]:35} | {article.metrics}")
        
        # Error analysis
        error_analysis = results['error_analysis']
        print(f"\nERROR ANALYSIS")
        print("-" * 40)
        print(f"Most common false positives:")
        for phrase, count in error_analysis['most_common_false_positives'][:5]:
            print(f"  • '{phrase}' ({count}x)")
        
        print(f"\nMost common false negatives:")
        for phrase, count in error_analysis['most_common_false_negatives'][:5]:
            print(f"  • '{phrase}' ({count}x)")
        
        # File output information
        if 'output_directory' in results['evaluation_config']:
            print(f"\nDETAILED RESULTS SAVED TO:")
            print(f"   {results['evaluation_config']['output_directory']}")
            print("   Files created:")
            print("   • detailed_comparison.csv - All predictions vs actual")
            print("   • article_results.csv - Per-article metrics")
            print("   • predicted_vs_actual_phrases.json - Detailed phrase lists")
            print("   • evaluation_summary.json - Complete summary")
            print("   • error_analysis.csv - Error patterns")


if __name__ == "__main__":
    # Demo usage
    evaluator = GoldStandardEvaluator()
    
    # Test single article evaluation (would need a trained model)
    print("Enhanced GoldStandardEvaluator initialized successfully!")
    print(f"Output directory: {evaluator.output_dir}")
    print(f"Stop headings filtered: {evaluator.stop_headings}")
    print("Use evaluator.evaluate_model(model, test_articles) to run evaluation.")