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
            image_parts = []
            
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
                    
                    # Prepare actual image for Gemini model
                    img_path = item.get("path")
                    if img_path and os.path.exists(img_path):
                        try:
                            import base64
                            from pathlib import Path
                            with open(img_path, "rb") as f:
                                img_bytes = f.read()
                            
                            ext = Path(img_path).suffix.lower()
                            mime_types = {
                                ".png": "image/png",
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg",
                                ".gif": "image/gif",
                                ".webp": "image/webp"
                            }
                            mime_type = mime_types.get(ext, "image/png")
                            
                            image_parts.append({
                                "mime_type": mime_type,
                                "data": base64.standard_b64encode(img_bytes).decode("utf-8")
                            })
                            context_parts.append(f"[Note: Image {i} pixel data is provided to you natively]")
                        except Exception as e:
                            print(f"   ⚠️  Could not load image {img_path} for generation: {e}")

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
- Actual image data provided inline

RULES:
1. Answer ONLY based on the provided document context and images — do NOT use outside knowledge
2. When referencing images, explicitly mention them: "As shown in Figure X on page Y..."
3. When referencing tables, mention them: "According to the table on page Z..."
4. Cite page numbers for all claims
5. If the answer is not in the context or visible in the provided images, clearly state: "This information is not available in the uploaded document"
6. Be precise with technical specifications, measurements, and engineering values
7. Look closely at the provided inline images to answer questions about charts, diagrams, or schematics"""

            user_prompt = f"""Question: {query}

Retrieved Document Content:
{context}

Please provide a comprehensive, technically accurate answer based ONLY on the above document content AND the provided images.
Include relevant page citations and reference any figures or tables mentioned.

After your answer, you MUST objectively evaluate the retrieval quality for this query by analyzing the retrieved chunks.
Count the following:
TP (True Positives): Number of retrieved sources/facts that were relevant and useful to answer the query.
FP (False Positives): Number of retrieved sources/facts that were NOT relevant or useful.
FN (False Negatives): Estimated number of missing facts/sources needed to fully answer the query but not retrieved.
(Note: Precision = TP / (TP + FP) and Recall = TP / (TP + FN))

Format your response exactly like this:
ANSWER:
[Your comprehensive answer here]
TP: [number]
FP: [number]
FN: [number]
PRECISION: [Score between 0.00 and 1.00]
RECALL: [Score between 0.00 and 1.00]"""

            print(f"📬 Sending to Gemini 2.5 Flash...")
            print(f"   📝 Text sources: {len(text_items)}")
            print(f"   📊 Table sources: {len(table_items)}")
            print(f"   🖼️  Image sources: {len(image_items)} (providing {len(image_parts)} raw images)")

            # Combine prompts and actual image bytes
            contents = [system_prompt, user_prompt]
            contents.extend(image_parts)

            response = self.model.generate_content(
                contents,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more factual answers
                    top_p=0.9,
                    max_output_tokens=2000
                )
            )

            import re
            
            answer_text = "No answer could be generated."
            precision_score = 0.0
            recall_score = 0.0
            tp_score = 0
            fp_score = 0
            fn_score = 0
            tn_score = 0  # we hardcode or ignore TN or default to 0 suitable for UI display

            if response.text:
                full_text = response.text
                
                # Extract TP
                tp_match = re.search(r"TP:\s*([0-9]+)", full_text)
                if tp_match:
                    try: tp_score = int(tp_match.group(1))
                    except: pass
                    
                # Extract FP
                fp_match = re.search(r"FP:\s*([0-9]+)", full_text)
                if fp_match:
                    try: fp_score = int(fp_match.group(1))
                    except: pass
                    
                # Extract FN
                fn_match = re.search(r"FN:\s*([0-9]+)", full_text)
                if fn_match:
                    try: fn_score = int(fn_match.group(1))
                    except: pass

                # Extract Precision
                prec_match = re.search(r"PRECISION:\s*([0-1]?\.?[0-9]+)", full_text)
                if prec_match:
                    try: precision_score = float(prec_match.group(1))
                    except ValueError: pass
                        
                # Extract Recall
                rec_match = re.search(r"RECALL:\s*([0-1]?\.?[0-9]+)", full_text)
                if rec_match:
                    try: recall_score = float(rec_match.group(1))
                    except ValueError: pass
                        
                # Extract Answer
                ans_match = re.search(r"ANSWER:\s*(.*?)(?=\nTP:|\nPRECISION:|$)", full_text, flags=re.DOTALL)
                if ans_match:
                    answer_text = ans_match.group(1).strip()
                else:
                    # fallback if regex fails
                    answer_text = full_text.split("TP:")[0].split("PRECISION:")[0].replace("ANSWER:", "").strip()

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
                "precision": precision_score,
                "recall": recall_score,
                "tp": tp_score,
                "fp": fp_score,
                "fn": fn_score,
                "tn": tn_score,
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
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "total_sources": 0,
                "modalities": {"text": 0, "tables": 0, "images": 0},
                "error": str(e)
            }


def generate_multimodal_answer(query: str, retrieved: Dict) -> Dict:
    """Main generation function"""
    generator = MultimodalGenerator()
    return generator.generate_answer(query, retrieved)
