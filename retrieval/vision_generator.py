"""
Gemini Vision API Integration
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
        self.model = self.genai.GenerativeModel('gemini-pro-vision')
        print("✅ Gemini Vision Generator initialized")

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
            return {"error": f"Image not found: {image_path}"}
        
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            # Determine image type
            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_types.get(ext, "image/jpeg")
            
            # Build prompt
            prompts = [
                "Provide a detailed but concise caption for this image (1-2 sentences).",
                "What are the main elements or information shown in this image?",
                "List key data points, equations, or relationships visible in the image.",
            ]
            
            if context:
                prompts[0] = f"Context: {context}\n" + prompts[0]
            
            results = {}
            
            for i, prompt in enumerate(prompts):
                try:
                    response = self.model.generate_content([
                        {
                            "role": "user",
                            "parts": [
                                {"inline_data": {"mime_type": mime_type, "data": image_data}},
                                prompt
                            ]
                        }
                    ])
                    
                    if response.text:
                        if i == 0:
                            results["caption"] = response.text.strip()
                        elif i == 1:
                            results["description"] = response.text.strip()
                        elif i == 2:
                            results["key_elements"] = response.text.strip()
                    
                except Exception as e:
                    print(f"⚠️  Error in analysis step {i}: {e}")
                    continue
            
            return {
                "image_path": image_path,
                "filename": Path(image_path).name,
                **results,
                "status": "success"
            }
        
        except Exception as e:
            print(f"❌ Error analyzing image {image_path}: {e}")
            return {
                "image_path": image_path,
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
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_types.get(ext, "image/jpeg")
            
            response = self.model.generate_content([
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": image_data}},
                        question
                    ]
                }
            ])
            
            return response.text if response.text else "No response generated"
        
        except Exception as e:
            return f"Error answering question: {e}"

    def analyze_multiple_images(
        self,
        image_paths: List[str],
        question: Optional[str] = None
    ) -> List[Dict]:
        """
        Analyze multiple images
        
        Args:
            image_paths: List of image paths
            question: Optional question to ask about all images
            
        Returns:
            List of analysis results
        """
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
            "caption": "Vision analysis not available",
            "description": "Please configure Gemini API for vision capabilities",
            "status": "unavailable"
        }

    def answer_image_question(self, image_path: str, question: str) -> str:
        """Fallback implementation"""
        return "Vision API not configured. Please set GEMINI_API_KEY environment variable."


def test_vision_generator():
    """Test Gemini Vision Generator"""
    try:
        generator = get_vision_generator()
        
        # Test with sample image if exists
        sample_img = "extracted_images/page_1_img_0.png"
        if os.path.exists(sample_img):
            print(f"🖼️  Analyzing: {sample_img}")
            analysis = generator.analyze_image(sample_img)
            
            if analysis.get("status") == "success":
                print(f"✅ Caption: {analysis.get('caption', 'N/A')}")
                print(f"📝 Description: {analysis.get('description', 'N/A')[:200]}")
                print(f"🔑 Key Elements: {analysis.get('key_elements', 'N/A')[:200]}")
            else:
                print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
        else:
            print(f"ℹ️  Sample image not found. Create it to test vision analysis.")
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Install: pip install google-generativeai")


if __name__ == "__main__":
    test_vision_generator()
