"""
Gemini Vision API Integration - FIXED
Uses gemini-2.5-flash (not deprecated gemini-pro-vision)
Understand and reason about images using Gemini's vision capabilities
"""

import os
import base64
from typing import List, Dict, Optional
from pathlib import Path


def get_vision_generator():
    """Get appropriate vision generator based on environment"""
    use_gemini = os.getenv("GENERATOR_TYPE", "gemini").lower() == "gemini"
    gemini_key = os.getenv("GEMINI_API_KEY")

    if use_gemini and gemini_key:
        return GeminiVisionGenerator(api_key=gemini_key)
    else:
        return VisionGeneratorFallback()


class GeminiVisionGenerator:
    """Generate captions and understanding of images using Gemini"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini Vision API"""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed. Install: pip install google-generativeai")

        self.genai = genai
        api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.genai.configure(api_key=api_key)
        # Use gemini-2.5-flash which supports vision - NOT the deprecated gemini-pro-vision
        self.model = self.genai.GenerativeModel("gemini-2.5-flash")
        print("✅ Gemini Vision Generator initialized (gemini-2.5-flash)")

    def analyze_image(
        self,
        image_path: str,
        context: Optional[str] = None
    ) -> Dict:
        """
        Analyze image and generate understanding

        Args:
            image_path: Path to image file
            context: Optional context about where image appears in document

        Returns:
            Dict with caption, description, and key elements
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}", "status": "error"}

        try:
            # Read image
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Determine image type
            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_types.get(ext, "image/png")

            # Build context-aware prompt
            context_str = f"\nContext: {context}" if context else ""

            prompt = f"""Analyze this image from an engineering/technical document.{context_str}

Please provide:
1. A concise caption (1-2 sentences describing what this image shows)
2. A detailed description of the key elements, data, or information visible
3. Any technical specifications, measurements, or important values visible

Format your response as:
CAPTION: [1-2 sentence caption]
DESCRIPTION: [detailed description]
KEY_ELEMENTS: [technical elements, values, relationships]"""

            # Use inline data for image
            image_part = {
                "mime_type": mime_type,
                "data": base64.standard_b64encode(image_bytes).decode("utf-8")
            }

            response = self.model.generate_content(
                contents=[image_part, prompt]
            )

            if not response.text:
                return {
                    "image_path": image_path,
                    "caption": "Image analysis unavailable",
                    "description": "No description generated",
                    "status": "error"
                }

            # Parse response
            response_text = response.text
            caption = ""
            description = ""
            key_elements = ""

            for line in response_text.split("\n"):
                line = line.strip()
                if line.startswith("CAPTION:"):
                    caption = line.replace("CAPTION:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("KEY_ELEMENTS:"):
                    key_elements = line.replace("KEY_ELEMENTS:", "").strip()

            # Fallback if formatting wasn't followed
            if not caption:
                caption = response_text[:200].strip()
            if not description:
                description = response_text.strip()

            return {
                "image_path": image_path,
                "filename": Path(image_path).name,
                "caption": caption or "Engineering diagram",
                "description": description,
                "key_elements": key_elements,
                "status": "success"
            }

        except Exception as e:
            print(f"❌ Error analyzing image {image_path}: {e}")
            return {
                "image_path": image_path,
                "caption": f"Image on page (analysis failed: {str(e)[:100]})",
                "description": "",
                "error": str(e),
                "status": "error"
            }

    def answer_image_question(
        self,
        image_path: str,
        question: str
    ) -> str:
        """
        Answer specific question about an image

        Args:
            image_path: Path to image
            question: Question about the image

        Returns:
            Answer from Gemini
        """
        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_types.get(ext, "image/png")

            image_part = {
                "mime_type": mime_type,
                "data": base64.standard_b64encode(image_bytes).decode("utf-8")
            }

            response = self.model.generate_content(
                contents=[image_part, f"Regarding this technical image: {question}"]
            )

            return response.text if response.text else "No response generated"

        except Exception as e:
            return f"Error answering question: {e}"

    def analyze_multiple_images(
        self,
        image_paths: List[str],
        question: Optional[str] = None
    ) -> List[Dict]:
        """Analyze multiple images"""
        results = []

        for img_path in image_paths:
            analysis = self.analyze_image(img_path)

            if question and analysis.get("status") == "success":
                analysis["answer"] = self.answer_image_question(img_path, question)

            results.append(analysis)

        return results


class VisionGeneratorFallback:
    """Fallback vision generator when Gemini not available"""

    def analyze_image(self, image_path: str, context: Optional[str] = None) -> Dict:
        """Fallback implementation"""
        return {
            "image_path": image_path,
            "filename": Path(image_path).name if image_path else "unknown",
            "caption": f"Figure from document: {Path(image_path).name if image_path else 'unknown'}",
            "description": "Vision analysis not available - configure GEMINI_API_KEY for detailed image analysis",
            "status": "unavailable"
        }

    def answer_image_question(self, image_path: str, question: str) -> str:
        """Fallback implementation"""
        return "Vision API not configured. Please set GEMINI_API_KEY environment variable."
