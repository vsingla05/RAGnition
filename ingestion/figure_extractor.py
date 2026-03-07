"""
Figure Extractor Module
Detects, extracts, and indexes images with captions and metadata from PDFs
"""

from typing import List, Dict, Any, Tuple, Optional
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FigureExtractor:
    """
    Extracts figures and images from PDF documents with captions and metadata
    """

    def __init__(self, pdf_path: str, text_content: str = ""):
        """
        Initialize figure extractor
        
        Args:
            pdf_path: Path to PDF file
            text_content: Full text content of PDF for caption extraction
        """
        self.pdf_path = pdf_path
        self.text_content = text_content
        self.figures = []

    def extract_figures(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract figures from images list and add metadata/captions
        
        Args:
            images: List of image objects from PDF extraction
                   Expected format: [{'page': int, 'image_index': int, 'data': bytes}]
        
        Returns:
            List of enriched figure objects with captions and metadata
        """
        enriched_figures = []

        for img_idx, image in enumerate(images):
            page_num = image.get('page', 0)
            
            # Extract caption for this image
            caption = self._extract_caption_for_image(page_num, img_idx)
            
            # Extract section context
            section = self._extract_section_for_page(page_num)
            
            # Create enriched figure object
            figure = {
                'id': f'fig_{page_num}_{img_idx}',
                'page': page_num,
                'index': img_idx,
                'caption': caption,
                'section': section,
                'type': self._determine_figure_type(caption),
                'description': self._generate_description(caption, section),
                'original_data': image.get('data'),
                'original_size': image.get('size', 0),
            }

            enriched_figures.append(figure)

        self.figures = enriched_figures
        return enriched_figures

    def _extract_caption_for_image(self, page_num: int, image_index: int) -> str:
        """
        Extract caption for image from surrounding text
        
        Args:
            page_num: Page number where image is located
            image_index: Index of image on page
            
        Returns:
            Extracted caption text, or default if none found
        """
        if not self.text_content:
            return f"Figure {page_num}.{image_index + 1}"

        # Look for common caption patterns near the image
        caption_patterns = [
            f'Figure {page_num}',
            f'Fig. {page_num}',
            f'Figure {page_num}.{image_index + 1}',
            'Figure',
            'Diagram',
        ]

        # Extract text for this page
        lines = self.text_content.split('\n')
        
        # Find line that mentions this figure
        for pattern in caption_patterns:
            for i, line in enumerate(lines):
                if pattern.lower() in line.lower():
                    # Return this line as caption
                    caption = line.strip()
                    if len(caption) > 5:  # Only if meaningful
                        return caption
        
        # Fallback caption
        return f"Figure {page_num}.{image_index + 1}"

    def _extract_section_for_page(self, page_num: int) -> str:
        """
        Extract the section heading for a given page
        
        Args:
            page_num: Page number
            
        Returns:
            Section name or empty string if not found
        """
        if not self.text_content:
            return ""

        lines = self.text_content.split('\n')
        
        # Look backwards from page content to find section header
        # This is a simplified approach - in production, use proper PDF structure
        section_keywords = ['Chapter', 'Section', 'Part', 'Module', 'System', 'Architecture']
        
        for keyword in section_keywords:
            for line in lines:
                if keyword in line and len(line.strip()) < 100:
                    return line.strip()
        
        return ""

    def _determine_figure_type(self, caption: str) -> str:
        """
        Determine the type of figure based on caption text
        
        Args:
            caption: Caption text
            
        Returns:
            Figure type: 'diagram', 'graph', 'table', 'screenshot', 'chart', 'other'
        """
        caption_lower = caption.lower()
        
        if any(word in caption_lower for word in ['diagram', 'flowchart', 'schematic', 'architecture']):
            return 'diagram'
        elif any(word in caption_lower for word in ['graph', 'chart', 'plot', 'curve']):
            return 'graph'
        elif any(word in caption_lower for word in ['table', 'comparison', 'matrix']):
            return 'table'
        elif any(word in caption_lower for word in ['screenshot', 'screen', 'window', 'ui']):
            return 'screenshot'
        elif any(word in caption_lower for word in ['plot', 'histogram', 'bar', 'pie']):
            return 'chart'
        else:
            return 'other'

    def _generate_description(self, caption: str, section: str) -> str:
        """
        Generate a detailed description for indexing
        
        Args:
            caption: Figure caption
            section: Section name
            
        Returns:
            Detailed description for embedding/indexing
        """
        parts = []
        
        if section:
            parts.append(f"From section: {section}")
        
        parts.append(f"Figure caption: {caption}")
        
        # Add figure type info
        fig_type = self._determine_figure_type(caption)
        type_descriptions = {
            'diagram': 'This is an architecture or system diagram showing components and relationships',
            'graph': 'This is a graph or chart displaying data visualization',
            'table': 'This is a table displaying structured data',
            'screenshot': 'This is a screenshot showing a user interface',
            'chart': 'This is a chart or plot showing quantitative information',
            'other': 'This is a figure referenced in the document',
        }
        
        if fig_type in type_descriptions:
            parts.append(type_descriptions[fig_type])
        
        return " | ".join(parts)

    def get_indexed_figures(self) -> List[Dict[str, Any]]:
        """
        Get figures ready for indexing into vector database
        
        Returns:
            List of figure objects with embeddings-ready descriptions
        """
        indexed = []
        
        for figure in self.figures:
            indexed.append({
                'id': figure['id'],
                'type': 'figure',
                'content': figure['description'],  # For embedding
                'metadata': {
                    'page': figure['page'],
                    'caption': figure['caption'],
                    'section': figure['section'],
                    'figure_type': figure['type'],
                    'source': 'figure_extraction',
                },
                'original_data': figure['original_data'],
            })
        
        return indexed

    def create_figure_text_references(self) -> List[Dict[str, Any]]:
        """
        Create text references that link figures to mentions in document text
        
        Returns:
            List of reference objects linking figures to text locations
        """
        references = []
        
        for figure in self.figures:
            # Find text mentions of this figure
            caption = figure['caption']
            section = figure['section']
            
            reference = {
                'figure_id': figure['id'],
                'caption': caption,
                'section': section,
                'page': figure['page'],
                'text_for_search': f"{caption} {section}",
                'link_text': f"See Figure {figure['page']}.{figure['index'] + 1}",
            }
            
            references.append(reference)
        
        return references

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of extracted figures
        
        Returns:
            Summary statistics
        """
        if not self.figures:
            return {
                'total_figures': 0,
                'by_type': {},
                'by_section': {},
            }

        by_type = {}
        by_section = {}
        
        for figure in self.figures:
            # Count by type
            fig_type = figure['type']
            by_type[fig_type] = by_type.get(fig_type, 0) + 1
            
            # Count by section
            section = figure['section'] or 'Unknown'
            by_section[section] = by_section.get(section, 0) + 1
        
        return {
            'total_figures': len(self.figures),
            'by_type': by_type,
            'by_section': by_section,
            'figures': [
                {
                    'id': f['id'],
                    'caption': f['caption'],
                    'page': f['page'],
                    'type': f['type'],
                }
                for f in self.figures[:10]  # First 10 for preview
            ],
        }


def process_pdf_for_figures(
    pdf_path: str,
    extracted_images: List[Dict[str, Any]],
    text_content: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    End-to-end figure processing pipeline
    
    Args:
        pdf_path: Path to PDF file
        extracted_images: Images extracted from PDF
        text_content: Full text content for caption extraction
        
    Returns:
        Tuple of (indexed_figures, summary)
    """
    extractor = FigureExtractor(pdf_path, text_content)
    
    # Extract and enrich figures
    figures = extractor.extract_figures(extracted_images)
    
    logger.info(f"Extracted {len(figures)} figures from {pdf_path}")
    
    # Get indexed version for vector database
    indexed_figures = extractor.get_indexed_figures()
    
    # Get summary
    summary = extractor.get_summary()
    
    return indexed_figures, summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock test
    mock_images = [
        {'page': 1, 'image_index': 0, 'size': 100000},
        {'page': 2, 'image_index': 0, 'size': 150000},
        {'page': 2, 'image_index': 1, 'size': 120000},
    ]
    
    mock_text = """
    Chapter 1: Introduction
    
    Figure 1.1 shows the system architecture
    Figure 1.2 displays the data flow diagram
    
    Chapter 2: System Design
    
    Figure 2.1: Architecture Overview
    This diagram illustrates the main components
    
    Figure 2.2: Component Relationships
    Shows how components interact
    """
    
    extractor = FigureExtractor("test.pdf", mock_text)
    figures = extractor.extract_figures(mock_images)
    
    print(f"Extracted {len(figures)} figures")
    print(extractor.get_summary())
