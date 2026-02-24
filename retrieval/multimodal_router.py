"""
Multimodal Query Router
Routes queries to appropriate retrieval pipeline based on detected intent
Handles: figures, tables, text, combined queries
"""

from typing import Dict, List, Tuple
from enum import Enum


class ModalityType(str, Enum):
    """Types of query modality"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    FIGURE = "figure"
    MIXED = "mixed"


class MultimodalQueryRouter:
    """Route queries to appropriate retrieval modality"""

    # Keywords indicating different modalities
    IMAGE_KEYWORDS = {
        "image", "figure", "photo", "picture", "diagram", "illustration",
        "screenshot", "visual", "show", "see", "look", "display", "plot",
        "graph", "chart", "visualization", "depicted", "shown"
    }

    TABLE_KEYWORDS = {
        "table", "data", "values", "numbers", "column", "row", "row",
        "spreadsheet", "matrix", "dataset", "statistics", "results",
        "comparison", "benchmark", "metric", "quantitative"
    }

    TEXT_KEYWORDS = {
        "explain", "what", "how", "why", "definition", "describe",
        "discuss", "state", "mention", "say", "write", "conclude",
        "abstract", "introduction", "section", "paragraph"
    }

    COMBINED_KEYWORDS = {
        "summarize", "overview", "comprehensive", "detailed", "all",
        "including", "together", "and", "both", "combine"
    }

    def __init__(self):
        self.modality_scores = {}

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine which modalities to retrieve
        
        Args:
            query: User query text
            
        Returns:
            Dict with:
            - primary_modality: Main modality to search
            - secondary_modalities: Additional modalities to include
            - confidence: Confidence score (0-1)
            - keywords_found: Keywords that triggered detection
        """
        query_lower = query.lower()
        tokens = set(query_lower.split())
        
        # Score each modality
        scores = {
            ModalityType.IMAGE: self._score_modality(tokens, self.IMAGE_KEYWORDS),
            ModalityType.TABLE: self._score_modality(tokens, self.TABLE_KEYWORDS),
            ModalityType.TEXT: self._score_modality(tokens, self.TEXT_KEYWORDS),
        }

        # Check for combined intent
        combined_score = self._score_modality(tokens, self.COMBINED_KEYWORDS)
        
        # Sort by score
        sorted_modalities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine primary and secondary
        primary = sorted_modalities[0][0]
        primary_score = sorted_modalities[0][1]

        secondary = []
        if combined_score > 0 or primary_score < 0.5:
            # If combined intent or low confidence, include all modalities
            secondary = [m[0] for m in sorted_modalities[1:] if m[1] > 0]
            overall_modality = ModalityType.MIXED
        else:
            secondary = [m[0] for m in sorted_modalities[1:] if m[1] > 0.3]
            overall_modality = primary

        # Collect keywords that triggered detection
        keywords_found = {
            "image": list(tokens & self.IMAGE_KEYWORDS),
            "table": list(tokens & self.TABLE_KEYWORDS),
            "text": list(tokens & self.TEXT_KEYWORDS),
            "combined": list(tokens & self.COMBINED_KEYWORDS)
        }

        return {
            "primary_modality": primary,
            "secondary_modalities": secondary,
            "overall_modality": overall_modality,
            "confidence": max(primary_score, combined_score),
            "modality_scores": scores,
            "combined_intent_score": combined_score,
            "keywords_found": keywords_found,
            "recommendation": self._get_recommendation(primary, secondary, combined_score)
        }

    def _score_modality(self, tokens: set, keywords: set) -> float:
        """Calculate modality score based on keyword matches"""
        if not keywords:
            return 0.0
        
        matches = len(tokens & keywords)
        return matches / len(keywords)  # Normalized score

    def _get_recommendation(self, primary: ModalityType, secondary: List[ModalityType], combined: float) -> str:
        """Get recommendation for retrieval strategy"""
        if combined > 0.3:
            return "Retrieve mixed modalities"
        elif primary == ModalityType.IMAGE:
            return "Search image collection for visual content"
        elif primary == ModalityType.TABLE:
            return "Search structured data and tables"
        elif primary == ModalityType.TEXT:
            return "Search text chunks with semantic search"
        else:
            return "Retrieve best matching content"

    def should_retrieve_modality(self, modality: ModalityType, analysis: Dict) -> bool:
        """Check if a modality should be retrieved"""
        if modality == analysis["overall_modality"]:
            return True
        
        if modality in analysis["secondary_modalities"]:
            return True
        
        # Always include text as fallback
        if modality == ModalityType.TEXT:
            return True
        
        return False

    def get_retrieval_strategy(self, analysis: Dict) -> Dict:
        """Get optimal retrieval strategy based on analysis"""
        strategy = {
            "modalities_to_search": [analysis["overall_modality"]],
            "include_secondary": False,
            "semantic_search_weight": 0.7,
            "keyword_search_weight": 0.3,
            "image_search_enabled": False,
            "table_search_enabled": False,
            "text_search_enabled": True,
        }

        # Enable secondary modalities if detected
        if analysis["secondary_modalities"]:
            strategy["include_secondary"] = True
            strategy["modalities_to_search"].extend(analysis["secondary_modalities"])

        # Adjust weights based on confidence
        if analysis["confidence"] < 0.4:
            strategy["semantic_search_weight"] = 0.5
            strategy["keyword_search_weight"] = 0.5

        # Set specific flags
        if analysis["overall_modality"] == ModalityType.IMAGE or ModalityType.IMAGE in analysis["secondary_modalities"]:
            strategy["image_search_enabled"] = True

        if analysis["overall_modality"] == ModalityType.TABLE or ModalityType.TABLE in analysis["secondary_modalities"]:
            strategy["table_search_enabled"] = True

        return strategy

    def format_analysis(self, analysis: Dict) -> str:
        """Format analysis as readable string"""
        output = []
        output.append(f"🎯 Query Analysis:")
        output.append(f"   Primary: {analysis['primary_modality'].value}")
        output.append(f"   Secondary: {[m.value for m in analysis['secondary_modalities']]}")
        output.append(f"   Confidence: {analysis['confidence']:.1%}")
        output.append(f"   Recommendation: {analysis['recommendation']}")
        
        if analysis['keywords_found']['image']:
            output.append(f"   Image keywords: {analysis['keywords_found']['image']}")
        if analysis['keywords_found']['table']:
            output.append(f"   Table keywords: {analysis['keywords_found']['table']}")
        
        return "\n".join(output)


def test_router():
    """Test query router"""
    router = MultimodalQueryRouter()
    
    test_queries = [
        "Show me the figure describing the architecture",
        "What data is in Table 3?",
        "Explain the methodology used in this paper",
        "Compare all the results in the paper including figures and tables",
        "What does the chart show?",
        "Can you describe all the visual elements?",
    ]
    
    print("📊 Query Router Tests:\n")
    
    for query in test_queries:
        analysis = router.analyze_query(query)
        print(f"Query: {query}")
        print(router.format_analysis(analysis))
        print()


if __name__ == "__main__":
    test_router()
