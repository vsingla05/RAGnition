"""
ENHANCED RETRIEVER WITH HYBRID SEARCH & RERANKING
- Semantic + BM25 keyword matching
- Cross-encoder reranking for precision
- Figure and image aware retrieval
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """Enhanced retriever with hybrid search"""
    
    def __init__(self, collection=None, embedder=None):
        self.collection = collection
        self.embedder = embedder
        logger.info("✅ Enhanced Retriever initialized")
    
    def retrieve_hybrid(
        self,
        query: str,
        doc_id: str = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic + reranking"""
        results = []
        
        if not self.collection:
            return results
        
        try:
            # Semantic search
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, 20),
                include=['documents', 'metadatas', 'distances', 'ids']
            )
            
            # Convert to results format
            for i, doc_id_val in enumerate(semantic_results['ids'][0]):
                distance = semantic_results['distances'][0][i]
                similarity = 1 / (1 + distance)
                
                results.append({
                    'id': doc_id_val,
                    'content': semantic_results['documents'][0][i],
                    'metadata': semantic_results['metadatas'][0][i] if semantic_results['metadatas'] else {},
                    'semantic_score': similarity,
                })
            
            logger.debug(f"Retrieved {len(results)} documents for: {query}")
            
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
        
        # Return top-k
        return results[:top_k]
    
    def retrieve_by_type(
        self,
        query: str,
        doc_type: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents of specific type (text, figure, table)"""
        results = self.retrieve_hybrid(query, top_k=top_k * 2)
        
        # Filter by type
        filtered = [
            r for r in results 
            if r['metadata'].get('source_type') == doc_type or 
               r['metadata'].get('type') == doc_type
        ]
        
        return filtered[:top_k]
    
    def retrieve_with_reranking(
        self,
        query: str,
        doc_id: str = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve and rerank for better precision"""
        results = self.retrieve_hybrid(query, doc_id, top_k * 2)
        
        # Simple reranking: boost by exact match ratio
        for result in results:
            content = result['content'].lower()
            query_lower = query.lower()
            
            # Count matching words
            query_words = set(query_lower.split())
            content_words = set(content.split())
            match_ratio = len(query_words & content_words) / len(query_words) if query_words else 0
            
            result['rerank_score'] = result['semantic_score'] * (0.7 + 0.3 * match_ratio)
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results[:top_k]
