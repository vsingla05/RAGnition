"""
ENHANCED CAPTION EXTRACTION & FIGURE LINKING
Extracts figure captions and associates them with images
Improves visual understanding by linking images to descriptive text
"""

import re
from typing import List, Dict, Tuple
from pathlib import Path


class CaptionExtractor:
    """Extract and link figure captions to images"""

    CAPTION_PATTERNS = [
        r"(?i)figure\s+(\d+)\s*[:\-]?\s*(.+?)(?=\n|figure|\Z)",
        r"(?i)fig\s*\.?\s*(\d+)\s*[:\-]?\s*(.+?)(?=\n|fig\.|\Z)",
        r"(?i)caption\s*[:\-]?\s*(.+?)(?=\n|figure|\Z)",
        r"(?i)image\s+(\d+)\s*[:\-]?\s*(.+?)(?=\n|image|\Z)",
    ]

    def __init__(self):
        self.captions = {}
        self.figure_numbers = set()

    def extract_captions_from_text(self, text: str, page_num: int) -> List[Dict]:
        """
        Extract captions from raw PDF text
        
        Returns:
        [
            {
                "figure_number": "3",
                "caption_text": "Rocket propulsion system diagram...",
                "page": 13,
                "confidence": 0.95,
                "location": "near_image"
            }
        ]
        """
        captions = []
        
        for pattern in self.CAPTION_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    if len(match.groups()) >= 2:
                        fig_num = match.group(1)
                        caption = match.group(2).strip()
                    else:
                        caption = match.group(1).strip()
                        fig_num = None
                    
                    if len(caption) > 10:  # Skip very short captions
                        captions.append({
                            "figure_number": fig_num,
                            "caption_text": caption[:500],  # Limit length
                            "page": page_num,
                            "confidence": 0.85,  # Pattern-based confidence
                            "location": "extracted_from_text",
                            "pattern": pattern
                        })
                        
                        if fig_num:
                            self.figure_numbers.add(fig_num)
                except Exception as e:
                    print(f"⚠️  Error extracting caption with pattern {pattern}: {e}")
        
        return captions

    def link_caption_to_image(self, 
                              caption_data: Dict, 
                              image_data: Dict,
                              max_page_distance: int = 2) -> bool:
        """
        Link caption to image based on:
        - Proximity (same page or adjacent pages)
        - Figure number matching
        - Content similarity
        
        Returns: True if link created
        """
        
        # Rule 1: Figure number match (highest confidence)
        caption_fig_num = caption_data.get("figure_number", "")
        image_fig_num = image_data.get("figure_number", "")
        
        if caption_fig_num and image_fig_num and caption_fig_num == image_fig_num:
            image_data["caption"] = caption_data["caption_text"]
            image_data["caption_source"] = "figure_number_match"
            image_data["has_caption"] = True
            return True
        
        # Rule 2: Proximity + contextual clues
        caption_page = caption_data.get("page", 0)
        image_page = image_data.get("page", 0)
        
        if abs(caption_page - image_page) <= max_page_distance:
            caption_text = caption_data.get("caption_text", "").lower()
            image_context = image_data.get("context", "").lower()
            
            # Check keyword overlap
            if image_context and len(caption_text) > 0:
                caption_words = set(caption_text.split())
                image_words = set(image_context.split())
                overlap = len(caption_words & image_words)
                
                if overlap >= 3:  # At least 3 words match
                    image_data["caption"] = caption_data["caption_text"]
                    image_data["caption_source"] = "proximity_match"
                    image_data["has_caption"] = True
                    return True
        
        return False

    def enhance_image_metadata(self, image_data: Dict, caption_data: Dict = None) -> Dict:
        """
        Enhance image metadata with caption and semantic information
        
        Returns enriched image metadata
        """
        
        enhanced = image_data.copy()
        
        if caption_data:
            enhanced.update({
                "caption": caption_data.get("caption_text", ""),
                "figure_number": caption_data.get("figure_number", ""),
                "caption_confidence": caption_data.get("confidence", 0.0),
                "has_caption": True,
                "semantic_type": self._infer_semantic_type(caption_data.get("caption_text", ""))
            })
        else:
            enhanced.update({
                "caption": "",
                "figure_number": "",
                "caption_confidence": 0.0,
                "has_caption": False,
                "semantic_type": "unknown"
            })
        
        return enhanced

    @staticmethod
    def _infer_semantic_type(caption_text: str) -> str:
        """
        Infer image type from caption text
        
        Returns: diagram, chart, graph, plot, architecture, flowchart, table, illustration, photo, etc.
        """
        
        caption_lower = caption_text.lower()
        
        type_keywords = {
            "diagram": ["diagram", "schematic", "layout"],
            "chart": ["chart", "graph", "plot"],
            "architecture": ["architecture", "system", "component"],
            "flowchart": ["flow", "process", "algorithm"],
            "table": ["table", "data", "results"],
            "illustration": ["illustration", "drawing", "sketch"],
            "screenshot": ["screenshot", "screen", "interface"],
            "photo": ["photo", "photograph", "image", "picture"],
        }
        
        for img_type, keywords in type_keywords.items():
            if any(kw in caption_lower for kw in keywords):
                return img_type
        
        return "diagram"  # Default assumption


class SectionDetector:
    """Detect document sections (intro, methods, results, appendix, etc.)"""

    SECTION_PATTERNS = {
        "introduction": r"(?i)^\s*(introduction|overview|background)\s*$",
        "methodology": r"(?i)^\s*(methods?|methodology|approach|procedure)\s*$",
        "results": r"(?i)^\s*(results?|findings?|output)\s*$",
        "discussion": r"(?i)^\s*(discussion|analysis)\s*$",
        "conclusion": r"(?i)^\s*(conclusion|summary|conclusion and future work)\s*$",
        "appendix": r"(?i)^\s*(appendix|appendices|supplementary materials?|supplemental)\s*$",
        "references": r"(?i)^\s*(references|bibliography|citations)\s*$",
        "abstract": r"(?i)^\s*(abstract|summary|executive summary)\s*$",
        "table_of_contents": r"(?i)^\s*(table of contents|contents|toc)\s*$",
    }

    def __init__(self):
        self.sections = []
        self.current_section = None

    def detect_sections(self, pdf_text_by_page: List[Tuple[int, str]]) -> List[Dict]:
        """
        Detect sections in PDF text
        
        Args:
            pdf_text_by_page: List of (page_number, text) tuples
        
        Returns:
            [
                {
                    "type": "introduction",
                    "start_page": 1,
                    "end_page": 5,
                    "title": "Introduction",
                    "is_appendix": False,
                    "hierarchy_level": 1
                }
            ]
        """
        
        sections = []
        current_section = None
        
        for page_num, text in pdf_text_by_page:
            lines = text.split('\n')
            
            for line in lines:
                # Check if line is a section header
                for section_type, pattern in self.SECTION_PATTERNS.items():
                    if re.match(pattern, line):
                        # Save previous section
                        if current_section:
                            current_section["end_page"] = page_num
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            "type": section_type,
                            "title": line.strip(),
                            "start_page": page_num,
                            "end_page": page_num,
                            "is_appendix": "appendix" in section_type.lower(),
                            "hierarchy_level": 1,  # Can be refined with heading analysis
                        }
                        break
        
        # Don't forget last section
        if current_section:
            sections.append(current_section)
        
        return sections

    def mark_sections_in_chunks(self, chunks: List[Dict], sections: List[Dict]) -> List[Dict]:
        """
        Add section metadata to chunks
        
        Returns enhanced chunks with section information
        """
        
        for chunk in chunks:
            page = chunk.get("page", 0)
            
            # Find which section this page belongs to
            for section in sections:
                if section["start_page"] <= page <= section["end_page"]:
                    chunk["section_type"] = section["type"]
                    chunk["section_title"] = section["title"]
                    chunk["is_appendix"] = section["is_appendix"]
                    chunk["hierarchy_level"] = section["hierarchy_level"]
                    break
            else:
                # No section found - mark as main content
                chunk["section_type"] = "main"
                chunk["section_title"] = "Main Content"
                chunk["is_appendix"] = False
                chunk["hierarchy_level"] = 0
        
        return chunks


class MetricKnowledgeBase:
    """
    Knowledge base for ML/AI metrics
    Helps disambiguate metric meaning and interpretation
    """
    
    METRICS = {
        "bleu": {
            "full_name": "BiLingual Evaluation Understudy",
            "category": "translation_quality",
            "task": "machine_translation",
            "range": [0, 100],
            "interpretation": "Higher is better",
            "unit": "percentage",
            "description": "Measures n-gram overlap between generated and reference translations"
        },
        "perplexity": {
            "full_name": "Perplexity",
            "category": "language_modeling",
            "task": "language_modeling",
            "range": [1, float('inf')],
            "interpretation": "Lower is better",
            "unit": "none",
            "description": "Average inverse probability of the test set under the model"
        },
        "rouge": {
            "full_name": "Recall-Oriented Understudy for Gisting Evaluation",
            "category": "summarization",
            "task": "text_summarization",
            "range": [0, 1],
            "interpretation": "Higher is better",
            "unit": "score",
            "description": "Measures overlap between generated and reference summaries"
        },
        "f1": {
            "full_name": "F1 Score",
            "category": "classification",
            "task": "classification",
            "range": [0, 1],
            "interpretation": "Higher is better",
            "unit": "score",
            "description": "Harmonic mean of precision and recall"
        },
        "accuracy": {
            "full_name": "Accuracy",
            "category": "classification",
            "task": "classification",
            "range": [0, 1],
            "interpretation": "Higher is better",
            "unit": "score",
            "description": "Proportion of correct predictions among all predictions"
        },
        "precision": {
            "full_name": "Precision",
            "category": "classification",
            "task": "classification",
            "range": [0, 1],
            "interpretation": "Higher is better",
            "unit": "score",
            "description": "Proportion of true positives among positive predictions"
        },
        "recall": {
            "full_name": "Recall",
            "category": "classification",
            "task": "classification",
            "range": [0, 1],
            "interpretation": "Higher is better",
            "unit": "score",
            "description": "Proportion of true positives among actual positives"
        },
    }

    @classmethod
    def get_metric_info(cls, metric_name: str) -> Dict:
        """Get information about a metric"""
        metric_lower = metric_name.lower().strip()
        return cls.METRICS.get(metric_lower, {
            "full_name": metric_name,
            "category": "unknown",
            "interpretation": "Unknown",
            "description": f"Metric: {metric_name}"
        })

    @classmethod
    def add_metric_context(cls, text: str) -> str:
        """
        Enhance text mentioning metrics with semantic context
        
        Transforms: "BLEU score is 35.2"
        Into: "BLEU score (machine translation quality metric, higher is better) is 35.2"
        """
        
        for metric_name in cls.METRICS:
            pattern = rf"\b{re.escape(metric_name)}\b"
            info = cls.get_metric_info(metric_name)
            
            replacement = f"{metric_name} ({info['category']}, {info['interpretation']})"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
