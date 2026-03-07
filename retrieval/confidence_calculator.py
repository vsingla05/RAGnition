"""
MULTI-FACTOR CONFIDENCE SCORING
Calibrates confidence based on:
- Retrieval quality (source relevance)
- Source coverage (how many sources support the answer)
- Semantic alignment (does answer align with sources?)
- Answer coherence (is answer internally consistent?)
- Grounding verification (answer backed by specific evidence)
"""

from typing import Dict, List
import re


class ConfidenceCalculator:
    """
    Calculate well-calibrated confidence scores
    
    Factors:
    1. Retrieval Quality (0.0-1.0): Average similarity of retrieved sources
    2. Source Coverage (0.0-1.0): Percentage of question covered by sources
    3. Semantic Alignment (0.0-1.0): Answer semantic similarity to sources
    4. Answer Coherence (0.0-1.0): Internal consistency of answer
    5. Grounding (0.0-1.0): Specific evidence references
    """
    
    def __init__(self):
        self.weights = {
            "retrieval_quality": 0.25,    # How good are the retrieved sources?
            "source_coverage": 0.25,      # How much of the question do sources cover?
            "semantic_alignment": 0.20,   # Does answer align with sources?
            "answer_coherence": 0.15,     # Is answer internally consistent?
            "grounding": 0.15,            # How well is answer grounded in sources?
        }
    
    def calculate_confidence(self,
                           query: str,
                           answer: str,
                           retrieved_sources: List[Dict],
                           answer_sources: List[str] = None) -> Dict:
        """
        Calculate multi-factor confidence score
        
        Args:
            query: Original question
            answer: Generated answer
            retrieved_sources: List of retrieved documents with similarity scores
            answer_sources: List of sources explicitly cited in answer
        
        Returns:
            {
                "overall_confidence": 0.0-1.0,
                "retrieval_quality": 0.0-1.0,
                "source_coverage": 0.0-1.0,
                "semantic_alignment": 0.0-1.0,
                "answer_coherence": 0.0-1.0,
                "grounding": 0.0-1.0,
                "reasoning": "Explanation of confidence score",
                "warnings": ["List of concerns"],
                "trust_level": "high/medium/low"
            }
        """
        
        # Factor 1: Retrieval Quality
        retrieval_quality = self._calculate_retrieval_quality(retrieved_sources)
        
        # Factor 2: Source Coverage
        source_coverage = self._calculate_source_coverage(query, retrieved_sources)
        
        # Factor 3: Semantic Alignment
        semantic_alignment = self._calculate_semantic_alignment(answer, retrieved_sources)
        
        # Factor 4: Answer Coherence
        answer_coherence = self._calculate_answer_coherence(answer)
        
        # Factor 5: Grounding
        grounding = self._calculate_grounding(answer, answer_sources or [])
        
        # Calculate weighted overall confidence
        overall_confidence = (
            self.weights["retrieval_quality"] * retrieval_quality +
            self.weights["source_coverage"] * source_coverage +
            self.weights["semantic_alignment"] * semantic_alignment +
            self.weights["answer_coherence"] * answer_coherence +
            self.weights["grounding"] * grounding
        )
        
        # Generate reasoning and warnings
        reasoning = self._generate_reasoning(
            retrieval_quality,
            source_coverage,
            semantic_alignment,
            answer_coherence,
            grounding
        )
        
        warnings = self._identify_warnings(
            overall_confidence,
            retrieval_quality,
            source_coverage,
            grounding,
            retrieved_sources
        )
        
        trust_level = self._determine_trust_level(overall_confidence, warnings)
        
        return {
            "overall_confidence": overall_confidence,
            "retrieval_quality": retrieval_quality,
            "source_coverage": source_coverage,
            "semantic_alignment": semantic_alignment,
            "answer_coherence": answer_coherence,
            "grounding": grounding,
            "reasoning": reasoning,
            "warnings": warnings,
            "trust_level": trust_level,
            "factor_breakdown": {
                "retrieval_quality": f"{retrieval_quality:.1%}",
                "source_coverage": f"{source_coverage:.1%}",
                "semantic_alignment": f"{semantic_alignment:.1%}",
                "answer_coherence": f"{answer_coherence:.1%}",
                "grounding": f"{grounding:.1%}",
            }
        }
    
    @staticmethod
    def _calculate_retrieval_quality(sources: List[Dict]) -> float:
        """
        Calculate retrieval quality based on source similarity scores
        
        High quality retrieval: avg similarity > 0.7
        Medium quality: 0.5-0.7
        Low quality: < 0.5
        """
        
        if not sources:
            return 0.0
        
        # Get similarity scores
        scores = []
        for source in sources:
            if "similarity" in source:
                scores.append(source["similarity"])
            elif "rerank_score" in source:
                # Normalize rerank score (typically -1 to 1)
                scores.append((source["rerank_score"] + 1) / 2)
            elif "ensemble_score" in source:
                scores.append(min(source["ensemble_score"], 1.0))
        
        if not scores:
            return 0.3  # Unknown retrieval quality
        
        avg_score = sum(scores) / len(scores)
        
        # Penalize if we have very few sources
        num_sources = len(sources)
        coverage_boost = min(1.0, num_sources / 5)  # More sources = more confidence
        
        quality = (avg_score * 0.8) + (coverage_boost * 0.2)
        
        return min(quality, 1.0)
    
    @staticmethod
    def _calculate_source_coverage(query: str, sources: List[Dict]) -> float:
        """
        Estimate what percentage of query is covered by sources
        
        Checks if key query concepts appear in retrieved sources
        """
        
        if not sources:
            return 0.0
        
        # Extract key concepts from query
        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 3)
        
        # Find which query words appear in sources
        source_text = " ".join([
            s.get("content", "").lower() for s in sources
        ])
        source_words = set(source_text.split())
        
        if not query_words:
            return 0.5
        
        # Coverage = what percentage of query concepts appear in sources
        covered_words = len(query_words & source_words)
        coverage = covered_words / len(query_words)
        
        return coverage
    
    @staticmethod
    def _calculate_semantic_alignment(answer: str, sources: List[Dict]) -> float:
        """
        Check if answer semantically aligns with sources
        
        Look for:
        - No contradictions with sources
        - Specific numbers/facts match sources
        - Claims are supported by sources
        """
        
        if not sources:
            return 0.3
        
        source_text = " ".join([s.get("content", "") for s in sources])
        
        # Extract numbers from answer and source
        answer_numbers = set(re.findall(r'\d+\.?\d*', answer))
        source_numbers = set(re.findall(r'\d+\.?\d*', source_text))
        
        # If answer has numbers, check if they appear in sources
        if answer_numbers:
            matching_numbers = len(answer_numbers & source_numbers)
            number_alignment = matching_numbers / len(answer_numbers)
        else:
            number_alignment = 0.7  # No numbers to verify
        
        # Check for contradiction indicators
        contradictions = 0
        contradiction_patterns = [
            (r"not ", r"is "),  # Negation mismatch
            (r"cannot", r"can "),
            (r"should not", r"should "),
        ]
        
        for neg_pattern, pos_pattern in contradiction_patterns:
            if re.search(neg_pattern, answer.lower()) and not re.search(neg_pattern, source_text.lower()):
                if re.search(pos_pattern, source_text.lower()):
                    contradictions += 1
        
        contradiction_penalty = min(contradictions * 0.2, 0.5)
        
        alignment = (number_alignment * 0.7) - contradiction_penalty
        
        return max(0.0, min(alignment, 1.0))
    
    @staticmethod
    def _calculate_answer_coherence(answer: str) -> float:
        """
        Measure answer coherence and consistency
        
        Checks:
        - Length (too short = low confidence)
        - Sentence structure (grammar)
        - Internal consistency (no self-contradictions)
        - Clarity (jargon vs plain language)
        """
        
        if not answer:
            return 0.0
        
        answer_lower = answer.lower()
        
        # Length check (reasonable answers are 50-2000 chars)
        length_score = 0.0
        if 50 <= len(answer) <= 2000:
            length_score = 1.0
        elif 30 <= len(answer) <= 3000:
            length_score = 0.7
        else:
            length_score = 0.3
        
        # Sentence count check
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_score = 1.0 if 2 <= len(sentences) <= 20 else 0.5
        
        # Self-contradiction check
        contradiction_words = ["but", "however", "on the other hand", "in contrast"]
        self_contradiction_score = 0.9 if any(cw in answer_lower for cw in contradiction_words) else 1.0
        
        # Combine
        coherence = (length_score * 0.4) + (sentence_score * 0.4) + (self_contradiction_score * 0.2)
        
        return min(coherence, 1.0)
    
    @staticmethod
    def _calculate_grounding(answer: str, answer_sources: List[str]) -> float:
        """
        Measure how well answer is grounded in specific sources
        
        Looks for:
        - Explicit citations ("As shown in...", "According to...")
        - Page references ("on page X")
        - Source references ("Figure 1", "Table 2")
        """
        
        if not answer:
            return 0.0
        
        answer_lower = answer.lower()
        
        # Citation patterns
        citation_patterns = [
            r"as (?:shown|described|stated|mentioned|illustrated) in",
            r"according to",
            r"page \d+",
            r"figure \d+",
            r"table \d+",
            r"section",
            r"source",
            r"document",
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, answer_lower))
        
        # Source references boost
        source_reference_boost = 0.0
        if answer_sources:
            # If explicit sources provided, boost confidence
            source_reference_boost = min(len(answer_sources) * 0.15, 0.5)
        
        # Grounding score based on citations
        citation_score = min(citation_count / 3, 1.0)  # 3+ citations = max score
        
        grounding = (citation_score * 0.6) + (source_reference_boost * 0.4)
        
        return min(grounding, 1.0)
    
    @staticmethod
    def _generate_reasoning(retrieval_q, coverage, alignment, coherence, grounding) -> str:
        """Generate human-readable explanation of confidence"""
        
        factors = [
            ("retrieval quality", retrieval_q),
            ("source coverage", coverage),
            ("semantic alignment", alignment),
            ("answer coherence", coherence),
            ("grounding", grounding),
        ]
        
        # Find strongest and weakest factors
        sorted_factors = sorted(factors, key=lambda x: x[1])
        weakest = sorted_factors[0]
        strongest = sorted_factors[-1]
        
        reasoning = f"Strong {strongest[0]} ({strongest[1]:.0%}). "
        
        if weakest[1] < 0.5:
            reasoning += f"Limited {weakest[0]} ({weakest[1]:.0%}) may reduce accuracy. "
        
        return reasoning
    
    @staticmethod
    def _identify_warnings(confidence: float, retrieval_q, coverage, grounding, sources) -> List[str]:
        """Identify concerns about confidence"""
        
        warnings = []
        
        if confidence < 0.5:
            warnings.append("⚠️  Low confidence - answer may be unreliable")
        
        if retrieval_q < 0.5:
            warnings.append("⚠️  Retrieved sources have low quality/relevance")
        
        if coverage < 0.5:
            warnings.append("⚠️  Sources may not fully address the question")
        
        if grounding < 0.3:
            warnings.append("⚠️  Answer lacks specific citations and references")
        
        if len(sources) < 2:
            warnings.append("⚠️  Only one or zero sources retrieved - consider as preliminary answer")
        
        return warnings
    
    @staticmethod
    def _determine_trust_level(confidence: float, warnings: List[str]) -> str:
        """Determine overall trust level"""
        
        if confidence >= 0.8 and len(warnings) == 0:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"


class AnswerValidator:
    """
    Validate answer against source context
    Ensures answer doesn't contradict sources
    """
    
    @staticmethod
    def validate_against_sources(answer: str, sources: List[Dict]) -> Dict:
        """
        Validate answer coherence with sources
        
        Returns:
        {
            "is_valid": bool,
            "contradiction_score": 0.0-1.0,  # Higher = more contradictions
            "issues": ["List of validation issues"]
        }
        """
        
        issues = []
        contradiction_score = 0.0
        
        source_text = " ".join([s.get("content", "") for s in sources])
        
        # Extract numeric claims from answer
        answer_numbers = re.findall(r'(\d+\.?\d*)\s+(\w+)', answer)
        source_numbers = re.findall(r'(\d+\.?\d*)\s+(\w+)', source_text)
        
        # Check for conflicting claims
        for ans_num, ans_unit in answer_numbers:
            conflicting = False
            for src_num, src_unit in source_numbers:
                if ans_unit.lower() == src_unit.lower() and ans_num != src_num:
                    conflicting = True
                    contradiction_score += 0.1
                    issues.append(f"Conflicting values for {ans_unit}: {ans_num} vs {src_num}")
            
            if not conflicting and (src_num, src_unit) not in source_numbers:
                # Number not found in sources
                issues.append(f"Value {ans_num} {ans_unit} not found in sources")
        
        is_valid = contradiction_score < 0.3 and len(issues) < 2
        
        return {
            "is_valid": is_valid,
            "contradiction_score": min(contradiction_score, 1.0),
            "issues": issues
        }
