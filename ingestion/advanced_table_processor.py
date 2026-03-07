"""
ADVANCED TABLE PROCESSING & STRUCTURED UNDERSTANDING
Parses tables into structured format with semantic metadata
Helps model understand table relationships, metrics, and column meanings
"""

import json
import re
from typing import Dict, List, Any
from pathlib import Path


class AdvancedTableProcessor:
    """
    Process tables with semantic understanding
    - Parse structure (rows, columns, headers)
    - Infer column semantics (metric, parameter, result)
    - Create multiple representations (text, json, markdown)
    - Add metric context
    """

    def __init__(self):
        self.tables_processed = 0
        self.metric_kb = self._load_metric_knowledge()

    def _load_metric_knowledge(self) -> Dict:
        """Load knowledge base of common metrics and their interpretations"""
        return {
            "bleu": {"type": "evaluation", "lower_is_better": False, "category": "translation"},
            "perplexity": {"type": "evaluation", "lower_is_better": True, "category": "language_model"},
            "rouge": {"type": "evaluation", "lower_is_better": False, "category": "summarization"},
            "f1": {"type": "evaluation", "lower_is_better": False, "category": "classification"},
            "accuracy": {"type": "evaluation", "lower_is_better": False, "category": "classification"},
            "precision": {"type": "evaluation", "lower_is_better": False, "category": "classification"},
            "recall": {"type": "evaluation", "lower_is_better": False, "category": "classification"},
            "loss": {"type": "training", "lower_is_better": True, "category": "training"},
            "auc": {"type": "evaluation", "lower_is_better": False, "category": "classification"},
            "map": {"type": "evaluation", "lower_is_better": False, "category": "ranking"},
        }

    def process_table(self, table_data: Dict, doc_id: str, doc_name: str, table_index: int) -> Dict:
        """
        Process table with semantic understanding
        
        Args:
            table_data: Raw table data from extractor
            doc_id: Document ID
            doc_name: Document name
            table_index: Table index in document
        
        Returns:
            Enhanced table with structured and text representations
        """
        
        try:
            page_num = table_data.get("page", "unknown")
            raw_dict = table_data.get("raw_dict", {})
            
            # Parse structure
            headers = self._extract_headers(raw_dict)
            rows = self._extract_rows(raw_dict)
            
            # Infer semantics
            column_types = self._infer_column_types(headers, rows)
            
            # Create multiple representations
            markdown_repr = self._create_markdown_table(headers, rows, page_num, table_index)
            json_repr = self._create_json_table(headers, rows, column_types)
            text_repr = self._create_text_summary(headers, rows, column_types, page_num)
            
            # Create semantic description
            semantic_desc = self._create_semantic_description(headers, rows, column_types)
            
            self.tables_processed += 1
            
            return {
                "text": markdown_repr,  # For vector DB storage
                "type": "table",
                "modality": "table",
                "metadata": {
                    "type": "table",
                    "modality": "table",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": int(page_num) if str(page_num).isdigit() else 0,
                    "table_index": table_index,
                    "summary": semantic_desc,
                    "format": "markdown",
                    "num_rows": len(rows),
                    "num_cols": len(headers),
                    "headers": headers,
                    "column_types": column_types,
                    "has_metrics": any(t == "metric" for t in column_types.values()),
                    "has_parameters": any(t == "parameter" for t in column_types.values()),
                },
                # Additional representations for enhanced retrieval
                "json_representation": json_repr,
                "text_summary": text_repr,
                "semantic_description": semantic_desc,
            }
        
        except Exception as e:
            print(f"⚠️  Error processing table {table_index}: {e}")
            return None

    def _extract_headers(self, raw_dict: Dict) -> List[str]:
        """Extract column headers from table"""
        try:
            if not raw_dict:
                return []
            
            # Try multiple header extraction strategies
            # Strategy 1: First row is typically headers
            if isinstance(raw_dict, dict) and "cells" in raw_dict:
                cells = raw_dict["cells"]
                if cells and len(cells) > 0:
                    # Find the topmost row
                    min_y = min(cell.get("y0", float('inf')) for cell in cells if cell)
                    headers = [cell.get("text", "") for cell in cells if cell and cell.get("y0", float('inf')) == min_y]
                    return headers
            
            # Strategy 2: Try treating first row as headers
            if isinstance(raw_dict, list) and len(raw_dict) > 0:
                return raw_dict[0]
            
            return []
        except Exception as e:
            print(f"   ⚠️  Error extracting headers: {e}")
            return []

    def _extract_rows(self, raw_dict: Dict) -> List[List[str]]:
        """Extract data rows from table"""
        try:
            if not raw_dict:
                return []
            
            # Similar to header extraction, but skip first row
            if isinstance(raw_dict, dict) and "cells" in raw_dict:
                cells = raw_dict["cells"]
                # Group cells by row (y-coordinate)
                rows_dict = {}
                for cell in cells:
                    if cell:
                        y = cell.get("y0", 0)
                        if y not in rows_dict:
                            rows_dict[y] = []
                        rows_dict[y].append(cell.get("text", ""))
                
                # Sort by y-coordinate and return
                sorted_rows = sorted(rows_dict.items())
                return [cells for _, cells in sorted_rows[1:]]  # Skip header row
            
            if isinstance(raw_dict, list) and len(raw_dict) > 1:
                return raw_dict[1:]
            
            return []
        except Exception as e:
            print(f"   ⚠️  Error extracting rows: {e}")
            return []

    def _infer_column_types(self, headers: List[str], rows: List[List[str]]) -> Dict[str, str]:
        """
        Infer column types (metric, parameter, model_name, result, etc.)
        
        Returns: {header: type, ...}
        """
        
        column_types = {}
        
        for idx, header in enumerate(headers):
            header_lower = header.lower().strip()
            
            # Check header keywords
            if any(kw in header_lower for kw in ["model", "method", "approach", "system"]):
                column_types[header] = "model_name"
            elif any(kw in header_lower for kw in ["bleu", "rouge", "f1", "accuracy", "precision", "recall", "auc", "loss", "perplexity"]):
                column_types[header] = "metric"
            elif any(kw in header_lower for kw in ["param", "lr", "batch", "epoch", "dropout", "hidden"]):
                column_types[header] = "parameter"
            elif any(kw in header_lower for kw in ["improvement", "gain", "delta", "difference", "change", "%"]):
                column_types[header] = "improvement"
            elif any(kw in header_lower for kw in ["config", "setting", "dataset", "train", "test"]):
                column_types[header] = "configuration"
            else:
                # Try to infer from data
                column_values = [row[idx] if idx < len(row) else "" for row in rows]
                column_types[header] = self._infer_value_type(column_values)
        
        return column_types

    @staticmethod
    def _infer_value_type(values: List[str]) -> str:
        """Infer type from actual values"""
        
        # Count numeric vs text
        numeric_count = 0
        for val in values:
            try:
                float(val.strip())
                numeric_count += 1
            except:
                pass
        
        # If mostly numeric, it's likely a metric/result
        if numeric_count / max(len(values), 1) > 0.7:
            return "numeric_value"
        
        # Check for common keywords
        text_content = " ".join(values).lower()
        if any(kw in text_content for kw in ["metric", "score", "loss"]):
            return "metric"
        
        return "text_value"

    def _create_markdown_table(self, headers: List[str], rows: List[List[str]], page_num: int, table_index: int) -> str:
        """Create markdown representation of table"""
        
        lines = [f"**Table {table_index} (Page {page_num})**\n"]
        
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return "\n".join(lines)

    @staticmethod
    def _create_json_table(headers: List[str], rows: List[List[str]], column_types: Dict[str, str]) -> str:
        """Create JSON representation of table"""
        
        table_json = {
            "headers": headers,
            "column_types": column_types,
            "rows": [dict(zip(headers, row)) for row in rows]
        }
        
        return json.dumps(table_json, indent=2)

    def _create_text_summary(self, headers: List[str], rows: List[List[str]], column_types: Dict[str, str], page_num: int) -> str:
        """Create text summary of table"""
        
        summary_parts = [f"Table on page {page_num} with {len(rows)} rows and {len(headers)} columns."]
        summary_parts.append(f"Columns: {', '.join(headers)}")
        
        # List column types
        metric_cols = [h for h, t in column_types.items() if t == "metric"]
        param_cols = [h for h, t in column_types.items() if t == "parameter"]
        
        if metric_cols:
            summary_parts.append(f"Metrics: {', '.join(metric_cols)}")
        if param_cols:
            summary_parts.append(f"Parameters: {', '.join(param_cols)}")
        
        # Add interpretation hints
        summary_parts.append("\n**Interpretation Guide:**")
        for col, col_type in column_types.items():
            if col in self.metric_kb:
                metric_info = self.metric_kb[col]
                summary_parts.append(f"- {col}: {metric_info['category']} metric, lower_is_better={metric_info['lower_is_better']}")
        
        return "\n".join(summary_parts)

    def _create_semantic_description(self, headers: List[str], rows: List[List[str]], column_types: Dict[str, str]) -> str:
        """Create semantic description that helps retrieval"""
        
        desc_parts = []
        
        # What are we comparing?
        model_cols = [h for h, t in column_types.items() if t == "model_name"]
        if model_cols:
            models = set()
            for row in rows:
                for col_idx, header in enumerate(headers):
                    if header in model_cols and col_idx < len(row):
                        models.add(row[col_idx])
            desc_parts.append(f"Compares models: {', '.join(models)}")
        
        # What are we measuring?
        metric_cols = [h for h, t in column_types.items() if t == "metric"]
        if metric_cols:
            desc_parts.append(f"Evaluates: {', '.join(metric_cols)}")
        
        # Under what conditions?
        config_cols = [h for h, t in column_types.items() if t == "configuration"]
        if config_cols:
            desc_parts.append(f"Configurations: {', '.join(config_cols)}")
        
        return " | ".join(desc_parts) if desc_parts else "Data table with structured results"


class TableRetrieval:
    """Enhanced retrieval for tables"""
    
    @staticmethod
    def create_table_query_context(table_metadata: Dict) -> str:
        """
        Create rich context for table query matching
        
        Helps retriever understand what the table is about
        """
        
        context_parts = []
        
        if "headers" in table_metadata:
            context_parts.append(f"Headers: {', '.join(table_metadata['headers'])}")
        
        if "summary" in table_metadata:
            context_parts.append(table_metadata["summary"])
        
        if "has_metrics" in table_metadata and table_metadata["has_metrics"]:
            context_parts.append("Contains evaluation metrics")
        
        if "column_types" in table_metadata:
            types = table_metadata["column_types"]
            metrics = [h for h, t in types.items() if t == "metric"]
            if metrics:
                context_parts.append(f"Metrics compared: {', '.join(metrics)}")
        
        return " | ".join(context_parts)
