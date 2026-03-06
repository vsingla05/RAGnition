"""
MULTIMODAL ANSWER GENERATOR - FIXED
Uses text, images, and tables to generate comprehensive answers
Uses Gemini 2.5 Flash with vision support
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))


class MultimodalGenerator:
    """Generate answers using all modalities"""

    def __init__(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed. Install: pip install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.genai = genai

    def generate_answer(self, query: str, retrieved: Dict) -> Dict:
        """
        Generate answer from multimodal retrieval results

        Args:
            query: User question
            retrieved: Output from retrieve_multimodal()

        Returns:
            {
                "answer": "...",
                "sources": [...],
                "images_referenced": [...],
                "confidence": 0.0-1.0
            }
        """

        print(f"\n🧠 GENERATING ANSWER")
        print("-" * 70)

        try:
            # Build context from all modalities
            context_parts = []
            images_to_reference = []

            # Add text results
            text_items = retrieved.get("text_results", [])
            if text_items:
                context_parts.append("\n📝 TEXT SOURCES:")
                for i, item in enumerate(text_items, 1):
                    context_parts.append(f"\n[Text {i}, Page {item.get('page', '?')}]")
                    context_parts.append(item.get("content", ""))

            # Add table results
            table_items = retrieved.get("table_results", [])
            if table_items:
                context_parts.append("\n\n📊 TABLE SOURCES:")
                for i, item in enumerate(table_items, 1):
                    context_parts.append(f"\n[Table {i}, Page {item.get('page', '?')}]")
                    context_parts.append(f"Summary: {item.get('summary', '')}")
                    context_parts.append(f"Content:\n{item.get('content', '')}")

            # Add image results with descriptions
            image_items = retrieved.get("image_results", [])
            if image_items:
                context_parts.append("\n\n🖼️  IMAGE SOURCES:")
                for i, item in enumerate(image_items, 1):
                    context_parts.append(f"\n[Image {i}, Page {item.get('page', '?')}]")
                    if item.get("caption"):
                        context_parts.append(f"Caption: {item.get('caption')}")
                    if item.get("description"):
                        context_parts.append(f"Description: {item.get('description')}")
                    if item.get("key_elements"):
                        context_parts.append(f"Key Elements: {item.get('key_elements')}")
                    images_to_reference.append(item)

            context = "\n".join(context_parts)

            if not context.strip():
                return {
                    "answer": "I couldn't find relevant information in the uploaded document for your question. Please try rephrasing or upload a more relevant document.",
                    "sources": [],
                    "images_referenced": [],
                    "confidence": 0.1,
                    "total_sources": 0,
                    "modalities": {"text": 0, "tables": 0, "images": 0}
                }

            # Build prompt with domain-aware instructions
            system_prompt = """You are an expert Engineering Document Intelligence System AI assistant. 
You analyze engineering manuals, technical documents, and specifications to provide precise answers.

You have access to:
- Text passages from the document
- Images with captions and descriptions (figures, diagrams, charts)
- Tables with summaries and content

RULES:
1. Answer ONLY based on the provided document context — do NOT use outside knowledge
2. When referencing images, explicitly mention them: "As shown in Figure X on page Y..."
3. When referencing tables, mention them: "According to the table on page Z..."
4. Cite page numbers for all claims
5. If the answer is not in the context, clearly state: "This information is not available in the uploaded document"
6. Be precise with technical specifications, measurements, and engineering values
7. If images contain diagrams or schematics, explain what they show"""

            user_prompt = f"""Question: {query}

Retrieved Document Content:
{context}

Please provide a comprehensive, technically accurate answer based ONLY on the above document content.
Include relevant page citations and reference any figures or tables mentioned."""

            print(f"📬 Sending to Gemini 2.5 Flash...")
            print(f"   📝 Text sources: {len(text_items)}")
            print(f"   📊 Table sources: {len(table_items)}")
            print(f"   🖼️  Image sources: {len(image_items)}")

            response = self.model.generate_content(
                [system_prompt, user_prompt],
                generation_config=self.genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more factual answers
                    top_p=0.9,
                    max_output_tokens=2000
                )
            )

            answer_text = response.text if response.text else "No answer could be generated."

            # Estimate confidence based on retrieval quality
            total_retrieved = retrieved.get("total_retrieved", 0)
            avg_similarity = 0.0
            all_sims = [
                r.get("similarity", 0) for r in text_items + image_items + table_items
                if r.get("similarity") is not None
            ]
            if all_sims:
                avg_similarity = sum(all_sims) / len(all_sims)

            confidence = min(0.95, 0.4 + (avg_similarity * 0.4) + (min(total_retrieved, 5) * 0.03))

            # Collect sources
            sources = []
            for item in text_items:
                sources.append({
                    "type": "text",
                    "page": item.get("page"),
                    "similarity": item.get("similarity"),
                    "content_preview": item.get("content", "")[:100]
                })
            for item in table_items:
                sources.append({
                    "type": "table",
                    "page": item.get("page"),
                    "similarity": item.get("similarity"),
                    "content_preview": item.get("summary", "")
                })
            for item in image_items:
                sources.append({
                    "type": "image",
                    "page": item.get("page"),
                    "similarity": item.get("similarity"),
                    "path": item.get("path"),
                    "content_preview": item.get("caption", "")
                })

            print(f"✅ Answer generated ({len(answer_text)} chars)")
            print(f"📚 Sources used: {len(sources)}")
            print(f"🎯 Confidence: {confidence:.2%}")

            # Convert image paths to URLs for frontend
            images_for_frontend = []
            for img in images_to_reference:
                img_copy = {
                    "page": img.get("page"),
                    "caption": img.get("caption", ""),
                    "description": img.get("description", ""),
                    "key_elements": img.get("key_elements", ""),
                    "filename": img.get("filename", ""),
                }
                # Build the URL
                filename = img.get("filename", "")
                path = img.get("path", "")
                if filename:
                    img_copy["path"] = f"/image/{filename}"
                    img_copy["url"] = f"http://localhost:8000/image/{filename}"
                elif path:
                    fn = Path(path).name
                    img_copy["path"] = f"/image/{fn}"
                    img_copy["url"] = f"http://localhost:8000/image/{fn}"
                images_for_frontend.append(img_copy)

            return {
                "answer": answer_text,
                "sources": sources,
                "images_referenced": images_for_frontend,
                "confidence": confidence,
                "total_sources": len(sources),
                "modalities": {
                    "text": len(text_items),
                    "tables": len(table_items),
                    "images": len(image_items)
                }
            }

        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "images_referenced": [],
                "confidence": 0.0,
                "total_sources": 0,
                "modalities": {"text": 0, "tables": 0, "images": 0},
                "error": str(e)
            }


def generate_multimodal_answer(query: str, retrieved: Dict) -> Dict:
    """Main generation function"""
    generator = MultimodalGenerator()
    return generator.generate_answer(query, retrieved)
