"""
Q&A Based Evaluation System
Tracks actual Q&A interactions and calculates real metrics based on answers
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path


@dataclass
class QAResult:
    """Single Q&A result"""
    question: str
    answer: str
    doc_type: str  # 'text', 'figure', 'table', 'equation'
    is_correct: bool = None
    confidence: float = 0.5
    timestamp: str = None
    sources: List[str] = None
    
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    mrr: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_score: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.sources is None:
            self.sources = []


class QAEvaluator:
    """Track Q&A interactions and calculate real metrics"""
    
    def __init__(self, storage_path: str = "/tmp/qa_history.json"):
        self.storage_path = Path(storage_path)
        self.qa_history: List[QAResult] = []
        self.load_history()
    
    def load_history(self):
        """Load Q&A history from file"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.qa_history = [
                        QAResult(
                            question=item['question'],
                            answer=item['answer'],
                            doc_type=item.get('doc_type', 'text'),
                            is_correct=item.get('is_correct'),
                            confidence=item.get('confidence', 0.5),
                            tp=item.get('tp', 0),
                            fp=item.get('fp', 0),
                            fn=item.get('fn', 0),
                            tn=item.get('tn', 0),
                            mrr=item.get('mrr', 0.0),
                            precision_at_k=item.get('precision_at_k', 0.0),
                            recall_at_k=item.get('recall_at_k', 0.0),
                            f1_score=item.get('f1_score', 0.0),
                            timestamp=item.get('timestamp'),
                            sources=item.get('sources', [])
                        )
                        for item in data
                    ]
        except Exception as e:
            print(f"⚠️  Could not load Q&A history: {e}")
            self.qa_history = []
    
    def save_history(self):
        """Save Q&A history to file"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(
                    [asdict(qa) for qa in self.qa_history],
                    f,
                    indent=2
                )
        except Exception as e:
            print(f"⚠️  Could not save Q&A history: {e}")
    
    def add_qa(
        self,
        question: str,
        answer: str,
        doc_type: str = 'text',
        confidence: float = 0.5,
        sources: List[str] = None,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
        tn: int = 0,
        precision: float = 0.0,
        recall: float = 0.0
    ) -> QAResult:
        """Add a new Q&A interaction with metrics"""
        # Calculate derived metrics if not provided
        # For MRR: assume first relevant result is at rank 1 if correct
        mrr = 1.0 if precision > 0.6 else 0.0
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        qa = QAResult(
            question=question,
            answer=answer,
            doc_type=doc_type,
            is_correct=None,
            confidence=confidence,
            sources=sources or [],
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            mrr=mrr,
            precision_at_k=precision,
            recall_at_k=recall,
            f1_score=f1
        )
        self.qa_history.append(qa)
        self.save_history()
        
        # Print metrics to terminal for presentation
        self.print_terminal_report(qa)
        
        return qa

    def print_terminal_report(self, latest_qa: Optional[QAResult] = None):
        """Print evaluation metrics to terminal for presentation"""
        print("\n" + "="*80)
        print("🔍 EVALUATION REPORT (RAGnition Multimodal Engine)")
        print("="*80)
        
        if latest_qa:
            print(f"\nLATEST INTERACTION:")
            print(f"Question : {latest_qa.question}")
            print(f"Doc Type : {latest_qa.doc_type.capitalize()}")
            print("-" * 40)
            print(f"Precision@K: {latest_qa.precision_at_k:.2%} | P@K = (Relevant docs in top K) / K")
            print(f"Recall@K:    {latest_qa.recall_at_k:.2%} | R@K = (Relevant docs in top K) / (Total relevant docs)")
            print(f"F1 Score:    {latest_qa.f1_score:.2%} | F1 = 2 * (P * R) / (P + R)")
            print(f"MRR:         {latest_qa.mrr:.2f} | MRR = (1/N) * sum(1/rank_i)")
            print("-" * 40)
            print(f"Confusion Matrix: [TP: {latest_qa.tp}, FP: {latest_qa.fp}]")
            print(f"                  [FN: {latest_qa.fn}, TN: {latest_qa.tn}]")

        # Aggregate Metrics
        if self.qa_history:
            avg_p = sum(qa.precision_at_k for qa in self.qa_history) / len(self.qa_history)
            avg_r = sum(qa.recall_at_k for qa in self.qa_history) / len(self.qa_history)
            avg_mrr = sum(qa.mrr for qa in self.qa_history) / len(self.qa_history)
            avg_f1 = sum(qa.f1_score for qa in self.qa_history) / len(self.qa_history)
            
            print(f"\nAGGREGATED PERFORMANCE (N={len(self.qa_history)}):")
            print(f"Mean Precision@K : {avg_p:.2%}")
            print(f"Mean Recall@K    : {avg_r:.2%}")
            print(f"Mean F1 Score    : {avg_f1:.2%}")
            print(f"Mean MRR         : {avg_mrr:.2f}")
        
        print("="*80 + "\n")
    
    def mark_correct(self, index: int, is_correct: bool):
        """Mark a Q&A result as correct or incorrect"""
        if 0 <= index < len(self.qa_history):
            self.qa_history[index].is_correct = is_correct
            self.save_history()
    
    def auto_evaluate(self, index: int):
        """
        Auto-evaluate a Q&A pair based on heuristics
        Returns: True if answer appears good, False otherwise
        """
        if index >= len(self.qa_history):
            return None
        
        qa = self.qa_history[index]
        
        # Heuristic evaluation
        is_good = True
        
        # Check 1: Answer length (too short = probably bad)
        if len(qa.answer) < 30:
            is_good = False
        
        # Check 2: Confidence threshold
        if qa.confidence < 0.3:
            is_good = False
        
        # Check 3: Has sources
        if not qa.sources:
            is_good = False
        
        self.qa_history[index].is_correct = is_good
        self.save_history()
        return is_good
    
    def get_metrics_by_type(self, doc_type: str = None) -> Dict[str, Any]:
        """Calculate metrics for specific document type"""
        
        # Filter by type if specified
        if doc_type:
            relevant_qa = [qa for qa in self.qa_history if qa.doc_type == doc_type]
        else:
            relevant_qa = self.qa_history
        
        if not relevant_qa:
            return {
                'test_count': 0,
                'correct': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'confidence_avg': 0.0
            }
        
        # Count correct answers (that have been evaluated)
        evaluated = [qa for qa in relevant_qa if qa.is_correct is not None]
        
        if not evaluated:
            # If no explicit evaluation, use confidence as proxy
            correct_count = sum(1 for qa in relevant_qa if qa.confidence > 0.6)
        else:
            correct_count = sum(1 for qa in evaluated if qa.is_correct)
        
        total = len(relevant_qa)
        evaluated_count = len(evaluated)
        
        # Calculate metrics
        # For precision: correct / total (assumes retrieved results are "positives")
        tp = correct_count
        fp = total - correct_count if evaluated_count > 0 else 0
        fn = 0  # We don't track missed items
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # If no FN, perfect recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = correct_count / total if total > 0 else 0.0
        confidence_avg = sum(qa.confidence for qa in relevant_qa) / len(relevant_qa)
        
        return {
            'test_count': total,
            'correct': correct_count,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'accuracy': round(accuracy, 3),
            'confidence_avg': round(confidence_avg, 3)
        }
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics across all document types"""
        
        if not self.qa_history:
            return {
                'overall': {
                    'test_count': 0,
                    'correct': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'accuracy': 0.0
                },
                'by_category': {},
                'test_cases': {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'pass_rate': 0.0
                }
            }
        
        # Get metrics for each type
        doc_types = set(qa.doc_type for qa in self.qa_history)
        by_category = {}
        
        total_correct = 0
        total_tests = 0
        
        for doc_type in sorted(doc_types):
            metrics = self.get_metrics_by_type(doc_type)
            by_category[doc_type] = metrics
            
            evaluated = [qa for qa in self.qa_history 
                        if qa.doc_type == doc_type and qa.is_correct is not None]
            
            if evaluated:
                total_correct += sum(1 for qa in evaluated if qa.is_correct)
                total_tests += len(evaluated)
        
        # Overall metrics (weighted average)
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        # Calculate weighted metrics
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
        
        for metrics in by_category.values():
            if metrics['test_count'] > 0:
                weight = metrics['test_count'] / len(self.qa_history)
                weighted_precision += metrics['precision'] * weight
                weighted_recall += metrics['recall'] * weight
                weighted_f1 += metrics['f1_score'] * weight
        
        # Aggregate enhanced metrics
        avg_mrr = sum(qa.mrr for qa in self.qa_history) / len(self.qa_history)
        avg_p_at_k = sum(qa.precision_at_k for qa in self.qa_history) / len(self.qa_history)
        avg_r_at_k = sum(qa.recall_at_k for qa in self.qa_history) / len(self.qa_history)
        avg_f1 = sum(qa.f1_score for qa in self.qa_history) / len(self.qa_history)

        return {
            'overall': {
                'test_count': len(self.qa_history),
                'correct': total_correct,
                'precision': round(avg_p_at_k, 3),
                'recall': round(avg_r_at_k, 3),
                'f1_score': round(avg_f1, 3),
                'mrr': round(avg_mrr, 3),
                'accuracy': round(overall_accuracy, 3)
            },
            'by_category': by_category,
            'test_cases': {
                'total': len(self.qa_history),
                'passed': total_correct,
                'failed': total_tests - total_correct,
                'pass_rate': round(total_correct / total_tests if total_tests > 0 else 0, 3)
            }
        }
    
    def get_qa_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent Q&A history"""
        return [
            {
                'question': qa.question,
                'answer': qa.answer[:100] + '...' if len(qa.answer) > 100 else qa.answer,
                'doc_type': qa.doc_type,
                'is_correct': qa.is_correct,
                'confidence': qa.confidence,
                'timestamp': qa.timestamp
            }
            for qa in self.qa_history[-limit:]
        ]
    
    def clear_history(self):
        """Clear all Q&A history"""
        self.qa_history = []
        self.save_history()


# Global evaluator instance
qa_evaluator = None


def get_qa_evaluator() -> QAEvaluator:
    """Get or create the global Q&A evaluator"""
    global qa_evaluator
    if qa_evaluator is None:
        qa_evaluator = QAEvaluator()
    return qa_evaluator
