"""
Multimodal Retrieval Pipeline
Orchestrates: text search + image search + table search + answer generation
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

from retrieval.multimodal_router import MultimodalQueryRouter, ModalityType
from retrieval.vision_generator import get_vision_generator
from retrieval.generator import generate_answer


class MultimodalRetrievalPipeline:
    """Full multimodal RAG orchestration"""

    def __init__(
        self,
        chroma_collection,
        text_embedder,
        image_embedder,
        top_k: int = 5,
        image_top_k: int = 3
    ):
        """
        Initialize multimodal retrieval pipeline
        
        Args:
            chroma_collection: Chroma collection for text retrieval
            text_embedder: Text embedding function
            image_embedder: Image embedding function
            top_k: Number of text results to retrieve
            image_top_k: Number of image results to retrieve
        """
        self.chroma_collection = chroma_collection
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.top_k = top_k
        self.image_top_k = image_top_k
        
        self.router = MultimodalQueryRouter()
        self.vision_gen = get_vision_generator()
        
        # Store all embeddings for similarity search
        self.all_embeddings = {}  # type: Dict[str, Dict]
        
        print("✅ Multimodal Retrieval Pipeline initialized")

    def add_document_data(self, metadata: Dict):
        """
        Register document data for retrieval
        
        Args:
            metadata: Dict with 'images', 'tables', 'text_chunks' keys
        """
        self.all_embeddings = metadata

    def retrieve(self, query: str) -> Dict:
        """
        Retrieve multimodal results for query
        
        Args:
            query: User query
            
        Returns:
            Dict with retrieved text, images, tables, and strategy used
        """
        # Step 1: Analyze query to determine modalities needed
        analysis = self.router.analyze_query(query)
        print(f"\n{self.router.format_analysis(analysis)}")
        
        # Step 2: Retrieve based on modality
        results = {
            "query": query,
            "query_analysis": analysis,
            "text_results": [],
            "image_results": [],
            "table_results": [],
            "combined_answer": None
        }

        # Retrieve text
        if self.router.should_retrieve_modality(ModalityType.TEXT, analysis):
            results["text_results"] = self._retrieve_text(query, self.top_k)

        # Retrieve images
        if self.router.should_retrieve_modality(ModalityType.IMAGE, analysis) or \
           self.router.should_retrieve_modality(ModalityType.FIGURE, analysis) or \
           self.router.should_retrieve_modality(ModalityType.CHART, analysis):
            results["image_results"] = self._retrieve_images(query, self.image_top_k)

        # Retrieve tables
        if self.router.should_retrieve_modality(ModalityType.TABLE, analysis):
            results["table_results"] = self._retrieve_tables(query)

        return results

    def _retrieve_text(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve text chunks using semantic search"""
        try:
            if not self.chroma_collection:
                return []
            
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            text_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    text_results.append({
                        "type": "text",
                        "content": doc,
                        "distance": results['distances'][0][i] if results.get('distances') else 0,
                        "score": 1 - (results['distances'][0][i] if results.get('distances') else 0)
                    })
            
            print(f"   📝 Retrieved {len(text_results)} text chunks")
            return text_results
        
        except Exception as e:
            print(f"   ⚠️  Error retrieving text: {e}")
            return []

    def _retrieve_images(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve images using multimodal search"""
        try:
            if not self.all_embeddings or 'images' not in self.all_embeddings:
                return []
            
            images = self.all_embeddings.get('images', [])
            if not images:
                return []
            
            # Get query embedding
            query_embedding = self.text_embedder([query])[0]
            
            # Calculate similarities
            similarities = []
            for img in images:
                if 'embedding' in img:
                    # Cosine similarity
                    import numpy as np
                    sim = float(np.dot(query_embedding, np.array(img['embedding'])))
                    similarities.append((img, sim))
            
            # Sort and get top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_images = similarities[:top_k]
            
            image_results = []
            for img_data, score in top_images:
                # Analyze image with vision model
                analysis = self.vision_gen.analyze_image(img_data['path'])
                
                image_results.append({
                    "type": "image",
                    "filename": img_data.get('filename', 'unknown'),
                    "path": img_data['path'],
                    "page": img_data.get('page', 'unknown'),
                    "similarity_score": score,
                    "caption": analysis.get('caption', 'No caption'),
                    "description": analysis.get('description', 'No description'),
                    "key_elements": analysis.get('key_elements', 'None identified')
                })
            
            print(f"   🖼️  Retrieved {len(image_results)} images")
            return image_results
        
        except Exception as e:
            print(f"   ⚠️  Error retrieving images: {e}")
            return []

    def _retrieve_tables(self, query: str) -> List[Dict]:
        """Retrieve tables using semantic search"""
        try:
            if not self.all_embeddings or 'tables' not in self.all_embeddings:
                return []
            
            tables = self.all_embeddings.get('tables', [])
            if not tables:
                return []
            
            # Get query embedding
            query_embedding = self.text_embedder([query])[0]
            
            # Calculate similarities
            similarities = []
            for table in tables:
                if 'embedding' in table:
                    import numpy as np
                    sim = float(np.dot(query_embedding, np.array(table['embedding'])))
                    similarities.append((table, sim))
            
            # Sort and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            table_results = []
            for table_data, score in similarities[:3]:
                table_results.append({
                    "type": "table",
                    "page": table_data.get('page', 'unknown'),
                    "content": str(table_data.get('content', '')),
                    "similarity_score": score
                })
            
            print(f"   📊 Retrieved {len(table_results)} tables")
            return table_results
        
        except Exception as e:
            print(f"   ⚠️  Error retrieving tables: {e}")
            return []

    def generate_multimodal_answer(
        self,
        query: str,
        retrieved: Dict,
        use_images: bool = True
    ) -> str:
        """
        Generate answer incorporating multimodal context
        
        Args:
            query: User query
            retrieved: Results from retrieve()
            use_images: Whether to use image analysis in answer
            
        Returns:
            Generated answer with visual context
        """
        # Build context from retrieved results
        context_parts = []
        
        # Add text context
        if retrieved['text_results']:
            context_parts.append("TEXT CONTEXT:")
            for result in retrieved['text_results'][:3]:
                context_parts.append(result['content'])
        
        # Add image context
        if use_images and retrieved['image_results']:
            context_parts.append("\nIMAGE CONTEXT:")
            for img in retrieved['image_results']:
                context_parts.append(f"Figure (Page {img['page']}):")
                context_parts.append(f"  Caption: {img['caption']}")
                context_parts.append(f"  Description: {img['description']}")
                context_parts.append(f"  Key elements: {img['key_elements']}")
        
        # Add table context
        if retrieved['table_results']:
            context_parts.append("\nTABLE CONTEXT:")
            for table in retrieved['table_results'][:2]:
                context_parts.append(f"Table (Page {table['page']}):")
                context_parts.append(table['content'][:500])
        
        context = "\n".join(context_parts)
        
        # Generate answer with full context
        answer = generate_answer(query, context)
        
        return answer

    def run_full_pipeline(self, query: str) -> Dict:
        """
        Run complete multimodal RAG pipeline
        
        Args:
            query: User query
            
        Returns:
            Complete results with answer and sources
        """
        print(f"\n🚀 Running multimodal RAG for: {query}")
        
        # Retrieve
        retrieved = self.retrieve(query)
        
        # Generate answer
        answer = self.generate_multimodal_answer(query, retrieved)
        
        # Compile sources
        sources = []
        for text in retrieved['text_results'][:2]:
            sources.append({"type": "text", "content": text['content'][:200]})
        for img in retrieved['image_results'][:2]:
            sources.append({"type": "image", "filename": img['filename'], "page": img['page']})
        for table in retrieved['table_results'][:1]:
            sources.append({"type": "table", "page": table['page']})
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "image_results": retrieved['image_results'],
            "text_results": retrieved['text_results'],
            "table_results": retrieved['table_results']
        }
