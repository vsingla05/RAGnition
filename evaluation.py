"""
Evaluation Module
Calculates precision, recall, F1 score, and accuracy metrics for RAG system
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float


class Evaluator:
    """
    Evaluates RAG system performance on various document types
    """

    def __init__(self):
        self.predictions = []
        self.ground_truth = []

    def calculate_precision(self, tp: int, fp: int) -> float:
        """
        Precision = TP / (TP + FP)
        What fraction of retrieved documents are relevant?
        """
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def calculate_recall(self, tp: int, fn: int) -> float:
        """
        Recall = TP / (TP + FN)
        What fraction of relevant documents were retrieved?
        """
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def calculate_f1(self, precision: float, recall: float) -> float:
        """
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Harmonic mean of precision and recall
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_accuracy(self, correct: int, total: int) -> float:
        """
        Accuracy = Correct / Total
        Fraction of correct predictions
        """
        if total == 0:
            return 0.0
        return correct / total

    def evaluate_qa_pair(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        answer_correct: bool,
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A pair
        
        Args:
            retrieved_docs: List of retrieved documents with 'doc_id' and 'relevance'
            relevant_doc_ids: List of document IDs that are actually relevant
            answer_correct: Whether the generated answer is correct
            
        Returns:
            Dictionary with TP, FP, FN, and answer correctness
        """
        retrieved_ids = {doc['doc_id'] for doc in retrieved_docs}
        relevant_ids = set(relevant_doc_ids)

        # True Positives: retrieved AND relevant
        tp = len(retrieved_ids & relevant_ids)

        # False Positives: retrieved but NOT relevant
        fp = len(retrieved_ids - relevant_ids)

        # False Negatives: relevant but NOT retrieved
        fn = len(relevant_ids - retrieved_ids)

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'answer_correct': answer_correct,
            'precision_at_k': self.calculate_precision(tp, fp),
            'recall_at_k': self.calculate_recall(tp, fn),
        }

    def evaluate_document_type(
        self,
        qa_pairs: List[Dict[str, Any]],
        doc_type: str,
    ) -> Dict[str, Any]:
        """
        Evaluate performance on a specific document type (text, figures, tables, etc.)
        
        Args:
            qa_pairs: List of Q&A evaluation results
            doc_type: Type of document ('text', 'figures', 'tables', 'equations')
            
        Returns:
            Metrics dictionary for this document type
        """
        if not qa_pairs:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
            }

        total_tp = sum(pair['tp'] for pair in qa_pairs)
        total_fp = sum(pair['fp'] for pair in qa_pairs)
        total_fn = sum(pair['fn'] for pair in qa_pairs)
        correct_answers = sum(1 for pair in qa_pairs if pair['answer_correct'])

        precision = self.calculate_precision(total_tp, total_fp)
        recall = self.calculate_recall(total_tp, total_fn)
        f1 = self.calculate_f1(precision, recall)
        accuracy = self.calculate_accuracy(correct_answers, len(qa_pairs))

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'test_count': len(qa_pairs),
            'correct': correct_answers,
        }

    def calculate_overall_metrics(
        self,
        by_category: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate overall metrics from per-category metrics
        
        Args:
            by_category: Dictionary of metrics by document type
            
        Returns:
            Overall metrics dictionary
        """
        # Average metrics across all categories (weighted by test count)
        total_tests = sum(metrics.get('test_count', 1) for metrics in by_category.values())
        
        if total_tests == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
            }

        weighted_precision = sum(
            metrics.get('precision', 0) * metrics.get('test_count', 1)
            for metrics in by_category.values()
        ) / total_tests

        weighted_recall = sum(
            metrics.get('recall', 0) * metrics.get('test_count', 1)
            for metrics in by_category.values()
        ) / total_tests

        weighted_accuracy = sum(
            metrics.get('accuracy', 0) * metrics.get('test_count', 1)
            for metrics in by_category.values()
        ) / total_tests

        f1 = self.calculate_f1(weighted_precision, weighted_recall)

        return {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': f1,
            'accuracy': weighted_accuracy,
        }

    def generate_evaluation_report(
        self,
        by_category: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate complete evaluation report
        
        Args:
            by_category: Dictionary of metrics by document type
            
        Returns:
            Complete evaluation report with overall and per-category metrics
        """
        overall = self.calculate_overall_metrics(by_category)

        # Calculate test statistics
        total_tests = sum(metrics.get('test_count', 1) for metrics in by_category.values())
        total_correct = sum(metrics.get('correct', 0) for metrics in by_category.values())
        pass_rate = total_correct / total_tests if total_tests > 0 else 0.0

        return {
            'overall': overall,
            'by_category': by_category,
            'test_cases': {
                'total': total_tests,
                'passed': total_correct,
                'failed': total_tests - total_correct,
                'pass_rate': pass_rate,
            },
        }


# Predefined test cases for evaluation
EVALUATION_TEST_CASES = {
    'text': [
        {
            'question': 'What is the system architecture?',
            'expected_sections': ['Section 2', 'Section 2.1'],
            'expected_answer': 'microservices',
        },
        {
            'question': 'Explain the API endpoints',
            'expected_sections': ['Section 3', 'API Documentation'],
            'expected_answer': 'RESTful',
        },
        {
            'question': 'What are the system requirements?',
            'expected_sections': ['Requirements', 'Installation'],
            'expected_answer': 'Python 3.8',
        },
    ],
    'figures': [
        {
            'question': 'Show the architecture diagram',
            'expected_figures': ['Figure 1', 'Figure 2.1'],
            'expected_answer': 'architecture',
        },
        {
            'question': 'What does the flow diagram show?',
            'expected_figures': ['Figure 3', 'Figure 3.2'],
            'expected_answer': 'data flow',
        },
        {
            'question': 'Describe the component relationships',
            'expected_figures': ['Figure 2', 'Figure 2.3'],
            'expected_answer': 'relationship',
        },
    ],
    'tables': [
        {
            'question': 'What are the performance metrics?',
            'expected_tables': ['Table 1', 'Performance Table'],
            'expected_answer': 'throughput',
        },
        {
            'question': 'Compare the different approaches',
            'expected_tables': ['Table 2', 'Comparison Table'],
            'expected_answer': 'comparison',
        },
        {
            'question': 'List the supported configurations',
            'expected_tables': ['Table 3', 'Configuration Table'],
            'expected_answer': 'configuration',
        },
    ],
    'equations': [
        {
            'question': 'What is the calculation formula?',
            'expected_equations': ['Equation 1', 'Formula 1'],
            'expected_answer': 'calculate',
        },
        {
            'question': 'Explain the mathematical model',
            'expected_equations': ['Equation 2', 'Formula 2'],
            'expected_answer': 'model',
        },
    ],
}


def get_mock_evaluation_report() -> Dict[str, Any]:
    """
    Generate a mock evaluation report for testing/demo purposes
    Shows realistic performance metrics for a RAG system
    """
    return {
        'overall': {
            'precision': 0.68,
            'recall': 0.95,
            'f1_score': 0.79,
            'accuracy': 0.78,
        },
        'by_category': {
            'text': {
                'precision': 0.92,
                'recall': 1.0,
                'f1_score': 0.96,
                'accuracy': 0.96,
                'test_count': 15,
                'correct': 15,
            },
            'figures': {
                'precision': 0.45,
                'recall': 0.65,
                'f1_score': 0.53,
                'accuracy': 0.50,
                'test_count': 10,
                'correct': 5,
            },
            'tables': {
                'precision': 0.75,
                'recall': 0.82,
                'f1_score': 0.78,
                'accuracy': 0.76,
                'test_count': 15,
                'correct': 12,
            },
            'equations': {
                'precision': 0.58,
                'recall': 0.72,
                'f1_score': 0.64,
                'accuracy': 0.62,
                'test_count': 10,
                'correct': 6,
            },
        },
        'test_cases': {
            'total': 50,
            'passed': 38,
            'failed': 12,
            'pass_rate': 0.76,
        },
    }
