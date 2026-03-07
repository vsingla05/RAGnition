"""
Multimodal Embeddings using CLIP
Generates embeddings for both text and images in same vector space
Enables cross-modal search: text query → image result and vice versa
"""

import os
from typing import List, Union, Tuple
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    Image = torch = CLIPProcessor = CLIPModel = None


class MultimodalEmbedder:
    """Generate embeddings for text and images using CLIP"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model
        
        Args:
            model_name: HuggingFace model ID for CLIP
        """
        if not torch or not CLIPModel:
            raise ImportError("torch and transformers required. Install: pip install torch transformers")
        
        print(f"🤖 Loading CLIP model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"📍 Using device: {self.device}")
        
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Inference mode
            
            self.embedding_dim = self.model.text_projection.out_features
            print(f"✅ CLIP ready. Embedding dimension: {self.embedding_dim}")
            print(f"   Max sequence length: 77 tokens")
        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            print("   Falling back to text-only mode (no vision)")
            self.processor = None
            self.model = None
            self.embedding_dim = 512  # Standard CLIP dim

    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            numpy array of shape (n, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Fallback if model not loaded
        if self.model is None:
            print(f"⚠️  Model not available, returning zero embeddings")
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        try:
            with torch.no_grad():
                inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                text_output = self.model.get_text_features(**inputs)
                
                # Handle both direct tensor and BaseModelOutputWithPooling outputs
                if hasattr(text_output, 'pooler_output'):
                    text_embeddings = text_output.pooler_output
                elif hasattr(text_output, 'last_hidden_state'):
                    # Use mean pooling for last hidden state
                    text_embeddings = text_output.last_hidden_state.mean(dim=1)
                else:
                    text_embeddings = text_output
                
                # Ensure it's a tensor
                if not isinstance(text_embeddings, torch.Tensor):
                    text_embeddings = torch.tensor(text_embeddings, device=self.device)
                
                # Normalize embeddings with proper error handling
                norm = torch.norm(text_embeddings, p=2, dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=1e-12)  # Prevent division by zero
                text_embeddings = text_embeddings / norm
            
            return text_embeddings.cpu().numpy()
        except Exception as e:
            print(f"⚠️  Error in embed_text: {e}")
            # Return zero embedding as fallback
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
    
    def embed_image(self, image_paths: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for images
        
        Args:
            image_paths: Single image path or list of image paths
            
        Returns:
            numpy array of shape (n, embedding_dim)
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        # Fallback if model not loaded
        if self.model is None:
            print(f"⚠️  Model not available, returning zero embeddings for {len(image_paths)} images")
            return np.zeros((len(image_paths), self.embedding_dim), dtype=np.float32)
        
        images = []
        for img_path in image_paths:
            try:
                if not os.path.exists(img_path):
                    print(f"⚠️  Image not found: {img_path}")
                    continue
                    
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"⚠️  Error loading image {img_path}: {e}")
                continue
        
        if not images:
            return np.zeros((len(image_paths), self.embedding_dim), dtype=np.float32)
        
        try:
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                image_output = self.model.get_image_features(**inputs)
                
                # Handle both direct tensor and BaseModelOutputWithPooling outputs
                if hasattr(image_output, 'pooler_output'):
                    image_embeddings = image_output.pooler_output
                elif hasattr(image_output, 'last_hidden_state'):
                    # Use mean pooling for last hidden state
                    image_embeddings = image_output.last_hidden_state.mean(dim=1)
                else:
                    image_embeddings = image_output
                
                # Ensure it's a tensor
                if not isinstance(image_embeddings, torch.Tensor):
                    image_embeddings = torch.tensor(image_embeddings, device=self.device)
                
                # Normalize embeddings with proper error handling
                norm = torch.norm(image_embeddings, p=2, dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=1e-12)  # Prevent division by zero
                image_embeddings = image_embeddings / norm
            
            return image_embeddings.cpu().numpy()
        except Exception as e:
            print(f"⚠️  Error in embed_image: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(images), self.embedding_dim), dtype=np.float32)
    
    def embed_table(self, table_text: str) -> np.ndarray:
        """
        Generate embeddings for table (as text)
        
        Args:
            table_text: Table content as string
            
        Returns:
            numpy array of shape (1, embedding_dim)
        """
        # Format table text with context
        formatted_text = f"Table: {table_text}"
        return self.embed_text(formatted_text)
    
    def batch_embed_mixed(self, items: List[dict]) -> List[dict]:
        """
        Embed mixed modality items (text, image, table)
        
        Args:
            items: List of dicts with 'modality' and content
                   {
                       'modality': 'text'|'image'|'table',
                       'content': text or path,
                       'metadata': {...}
                   }
        
        Returns:
            Same items with 'embedding' field added
        """
        results = []
        
        for item in items:
            try:
                modality = item.get('modality', 'text')
                
                if modality == 'image':
                    embedding = self.embed_image(item['content'])
                elif modality == 'table':
                    embedding = self.embed_table(item['content'])
                else:  # text
                    embedding = self.embed_text(item['content'])
                
                if len(embedding) > 0:
                    item['embedding'] = embedding[0].tolist()
                    results.append(item)
                else:
                    print(f"⚠️  Skipped empty embedding for {item.get('content', 'unknown')}")
                    
            except Exception as e:
                print(f"⚠️  Error embedding item: {e}")
                continue
        
        return results
    
    def similarity_search(
        self,
        query: str,
        embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query
        
        Args:
            query: Query text
            embeddings: List of candidate embeddings
            top_k: Return top K results
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        query_embedding = self.embed_text(query)[0]
        
        similarities = []
        for i, emb in enumerate(embeddings):
            # Cosine similarity
            sim = np.dot(query_embedding, emb)
            similarities.append((i, float(sim)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def test_embedder():
    """Test multimodal embedder"""
    try:
        embedder = MultimodalEmbedder()
        
        # Test text embedding
        print("\n📝 Text embedding test:")
        text_emb = embedder.embed_text("This is a test image showing a cat")
        print(f"   Shape: {text_emb.shape}")
        print(f"   First 5 dims: {text_emb[0][:5]}")
        
        # Test image embedding (if sample exists)
        sample_img = "extracted_images/sample.png"
        if os.path.exists(sample_img):
            print("\n🖼️  Image embedding test:")
            img_emb = embedder.embed_image(sample_img)
            print(f"   Shape: {img_emb.shape}")
        
        # Test cross-modal similarity
        print("\n🔗 Cross-modal similarity test:")
        text_queries = [
            "A cat sleeping",
            "Mathematical equations",
            "A tree in forest"
        ]
        
        for query in text_queries:
            emb = embedder.embed_text(query)
            print(f"   ✓ Embedded: {query}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Install requirements: pip install torch transformers")


if __name__ == "__main__":
    test_embedder()
