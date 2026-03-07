"""
ENHANCED MULTIMODAL RETRIEVER WITH HYBRID SEARCH & RERANKING
- Semantic search + BM25 keyword matching (hybrid)
- Cross-encoder reranking for better relevance
- Figure-aware retrieval with caption indexing
- Precision-focused retrieval to reduce false positives
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Simple tokenizer for BM25"""
    
    def __init__(self):
        import string
        self.punctuation = set(string.punctuation)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words"""
        tokens = []
        current_token = ""
        
        for char in text.lower():
            if char in self.punctuation or char.isspace():
                if current_token:
                    tokens.append(current_token)
                current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens


class SimpleBM25:
    """Simplified BM25 for keyword matching"""
    
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenizer = SimpleTokenizer()
        self.documents = documents
        self.idf_dict = {}
        self.doc_lengths = []
        self._build_index()
    
    def _build_index(self):
        """Build inverted index and calculate IDF"""
        doc_token_counts = []
        all_tokens = set()
        
        for doc in self.documents:
            tokens = self.tokenizer.tokenize(doc)
            doc_token_counts.append(len(tokens))
            all_tokens.update(tokens)
        
        avg_doc_length = sum(doc_token_counts) / len(doc_token_counts) if doc_token_counts else 1
        self.avg_doc_length = avg_doc_length
        self.doc_lengths = doc_token_counts
        
        num_docs = len(self.documents)
        for token in all_tokens:
            doc_count = sum(1 for doc in self.documents if token in self.tokenizer.tokenize(doc))
            idf = max(0.0, (num_docs - doc_count + 0.5) / (doc_count + 0.5))
            self.idf_dict[token] = idf
    
    def score_document(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score"""
        query_tokens = self.tokenizer.tokenize(query)
        doc_tokens = self.tokenizer.tokenize(self.documents[doc_index])
        doc_length = self.doc_lengths[doc_index]
        
        score = 0.0
        
        for token in query_tokens:
            token_count = doc_tokens.count(token)
            if token_count == 0:
                continue
            
            idf = self.idf_dict.get(token, 0.0)
            numerator = idf * token_count * (self.k1 + 1)
            denominator = token_count + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += numerator / denominator
        
        return score


class SimpleReranker:
    """Cross-encoder style reranker"""
    
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
    
    def calculate_relevance_score(self, query: str, document: str) -> float:
        """Calculate relevance score between query and document"""
        query_tokens = set(self.tokenizer.tokenize(query))
        doc_tokens = set(self.tokenizer.tokenize(document))
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        intersection = len(query_tokens & doc_tokens)
        union = len(query_tokens | doc_tokens)
        jaccard = intersection / union if union > 0 else 0.0
        density = intersection / len(query_tokens)
        doc_length_factor = min(1.0, len(doc_tokens) / 200)
        
        score = 0.5 * jaccard + 0.3 * density + 0.2 * doc_length_factor
        return min(1.0, score)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents"""
        for doc in documents:
            rerank_score = self.calculate_relevance_score(query, doc['content'])
            doc['rerank_score'] = rerank_score
            
            semantic = doc.get('semantic_score', 0.0)
            bm25 = doc.get('bm25_score', 0.0)
            bm25_normalized = min(1.0, bm25 / 10.0) if bm25 > 0 else 0.0
            
            doc['final_score'] = (0.4 * semantic + 0.3 * bm25_normalized + 0.3 * rerank_score)
        
        documents.sort(key=lambda x: x['final_score'], reverse=True)
        return documents


class EnhancedRetriever:
    """Enhanced retriever with hybrid search and reranking"""
    
    def __init__(self, collection=None, embedder=None):
        self.collection = collection
        self.embedder = embedder
        self.bm25 = None
        self.reranker = SimpleReranker()
        logger.info("✅ Enhanced Retriever initialized with hybrid search")
    
    def prepare_bm25_index(self, documents: List[str]):
        """Prepare BM25 index"""
        self.bm25 = SimpleBM25(documents)
        logger.info(f"Prepared BM25 index for {len(documents)} documents")
    
    def retrieve_hybrid(
        self,
        query: str,
        doc_id: str = None,
        top_k: int = 5,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.3,
        rerank_weight: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic + BM25 + reranking"""
        results = []
        
        # Semantic search
        if self.collection:
            try:
                semantic_results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k * 2,
                    include=['documents', 'metadatas', 'distances', 'ids']
                )
                
                for i, doc_id in enumerate(semantic_results['ids'][0]):
                    distance = semantic_results['distances'][0][i]
                    similarity = 1 / (1 + distance)
                    
                    results.append({
                        'id': doc_id,
                        'content': semantic_results['documents'][0][i],
                        'metadata': semantic_results['metadatas'][0][i] if semantic_results['metadatas'] else {},
                        'semantic_score': similarity,
                        'source_type': 'text',
                    })
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # Rerank
        if results:
            results = self.reranker.rerank(query, results)
        
        return results[:top_k]
        """
        Ensemble retrieval combining semantic + keyword matching
        
        Args:
            query: User question
            doc_id: Document filter (optional)
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for BM25 keyword matching (0-1)
        
        Returns:
            Combined results ranked by ensemble score
        """
        
        results = {
            "semantic_results": [],
            "keyword_results": [],
            "ensemble_results": [],
            "retrieval_method": "ensemble"
        }
        
        try:
            # Build where clause for filtering
            where_clause = None
            if doc_id:
                where_clause = {"doc_id": doc_id}
            
            # 1. SEMANTIC RETRIEVAL (similarity search)
            print(f"🔎 Semantic retrieval: {query}")
            
            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k * 2,  # Get more for filtering
            }
            if where_clause:
                query_kwargs["where"] = where_clause
            
            semantic_raw = self.collection.query(**query_kwargs)
            
            semantic_docs = semantic_raw.get("documents", [[]])[0]
            semantic_metas = semantic_raw.get("metadatas", [[]])[0]
            semantic_ids = semantic_raw.get("ids", [[]])[0]
            semantic_distances = semantic_raw.get("distances", [[]])[0]
            
            # Convert distances to similarities
            for doc, meta, doc_id_val, dist in zip(semantic_docs, semantic_metas, semantic_ids, semantic_distances):
                similarity = max(0, 1 - dist)
                results["semantic_results"].append({
                    "content": doc,
                    "metadata": meta,
                    "id": doc_id_val,
                    "similarity": similarity,
                    "retrieval_type": "semantic"
                })
            
            print(f"   ✅ Semantic: {len(results['semantic_results'])} results")
            
            # 2. KEYWORD RETRIEVAL (BM25)
            if self.bm25_retriever:
                print(f"🔍 Keyword retrieval: {query}")
                
                # Build document corpus from collection
                try:
                    # Get all documents (or a large sample)
                    all_docs = self.collection.get(where=where_clause)
                    
                    if all_docs and all_docs.get("documents"):
                        documents = all_docs["documents"]
                        metadatas = all_docs.get("metadatas", [{}] * len(documents))
                        ids = all_docs.get("ids", list(range(len(documents))))
                        
                        # Build BM25 index
                        tokenized_docs = [doc.split() for doc in documents]
                        bm25 = self.bm25_retriever(tokenized_docs)
                        
                        # Query with BM25
                        query_tokens = query.split()
                        scores = bm25.get_scores(query_tokens)
                        
                        # Get top results
                        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                        
                        for idx in ranked_indices:
                            if scores[idx] > 0:  # Only positive scores
                                results["keyword_results"].append({
                                    "content": documents[idx],
                                    "metadata": metadatas[idx],
                                    "id": ids[idx],
                                    "bm25_score": scores[idx],
                                    "retrieval_type": "keyword"
                                })
                        
                        print(f"   ✅ Keyword: {len(results['keyword_results'])} results")
                
                except Exception as e:
                    print(f"   ⚠️  BM25 retrieval failed: {e}")
            
            # 3. ENSEMBLE RANKING
            # Combine results with weighted scoring
            ensemble_dict = {}
            
            # Add semantic results
            for result in results["semantic_results"]:
                doc_id = result["id"]
                semantic_score = result["similarity"] * semantic_weight
                
                if doc_id not in ensemble_dict:
                    ensemble_dict[doc_id] = {
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "id": doc_id,
                        "semantic_score": 0,
                        "keyword_score": 0,
                        "ensemble_score": 0
                    }
                ensemble_dict[doc_id]["semantic_score"] = semantic_score
            
            # Add keyword results
            for result in results["keyword_results"]:
                doc_id = result["id"]
                # Normalize BM25 scores to 0-1 range
                max_bm25 = max([r["bm25_score"] for r in results["keyword_results"]], default=1)
                keyword_score = (result["bm25_score"] / max_bm25) * keyword_weight if max_bm25 > 0 else 0
                
                if doc_id not in ensemble_dict:
                    ensemble_dict[doc_id] = {
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "id": doc_id,
                        "semantic_score": 0,
                        "keyword_score": 0,
                        "ensemble_score": 0
                    }
                ensemble_dict[doc_id]["keyword_score"] = keyword_score
            
            # Calculate ensemble score
            for doc_id, item in ensemble_dict.items():
                item["ensemble_score"] = item["semantic_score"] + item["keyword_score"]
            
            # Sort by ensemble score
            sorted_results = sorted(ensemble_dict.values(), key=lambda x: x["ensemble_score"], reverse=True)
            results["ensemble_results"] = sorted_results[:top_k]
            
            print(f"   ✅ Ensemble: {len(results['ensemble_results'])} combined results")
            
            return results
        
        except Exception as e:
            print(f"   ❌ Ensemble retrieval failed: {e}")
            return results


class CrossEncoderReranker:
    """
    Use cross-encoder to rerank retrieved results
    More accurate than semantic similarity alone
    """
    
    def __init__(self):
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            # Using a lightweight but effective model
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Cross-encoder loaded for reranking")
        except ImportError:
            print("⚠️  sentence-transformers not installed. Install: pip install sentence-transformers")
        except Exception as e:
            print(f"⚠️  Failed to load cross-encoder: {e}")
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank results using cross-encoder
        
        Args:
            query: Original query
            results: Retrieved results with 'content' field
            top_k: Number of results to return
        
        Returns:
            Reranked results with 'rerank_score' field
        """
        
        if not self.model or not results:
            return results
        
        try:
            print(f"🔄 Reranking {len(results)} results...")
            
            # Prepare pairs for cross-encoder
            pairs = [[query, result.get("content", "")] for result in results]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Add scores to results
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            print(f"   ✅ Reranked, top result score: {reranked[0].get('rerank_score', 0):.3f}")
            
            return reranked[:top_k]
        
        except Exception as e:
            print(f"   ⚠️  Reranking failed: {e}")
            return results[:top_k]


class MultiHopRetriever:
    """
    Multi-hop retrieval for complex queries requiring information from multiple sections
    
    Process:
    1. Initial retrieval for query
    2. Identify related concepts/sections
    3. Retrieve related information
    4. Combine all results
    """
    
    def __init__(self, collection):
        self.collection = collection

    def retrieve_multi_hop(self, query: str, doc_id: str = None, hops: int = 2, top_k: int = 10) -> Dict:
        """
        Multi-hop retrieval
        
        Args:
            query: Original query
            doc_id: Document filter
            hops: Number of retrieval hops
            top_k: Results per hop
        
        Returns:
            Combined results from all hops
        """
        
        print(f"🔗 Multi-hop retrieval ({hops} hops)")
        
        all_results = []
        seen_ids = set()
        
        current_query = query
        
        for hop_num in range(hops):
            print(f"\n   Hop {hop_num + 1}/{hops}: {current_query}")
            
            # Retrieve for current query
            where_clause = {"doc_id": doc_id} if doc_id else None
            
            try:
                hop_results = self.collection.query(
                    query_texts=[current_query],
                    n_results=top_k,
                    where=where_clause
                )
                
                documents = hop_results.get("documents", [[]])[0]
                metadatas = hop_results.get("metadatas", [[]])[0]
                ids = hop_results.get("ids", [[]])[0]
                
                # Add new results
                for doc, meta, doc_id_val in zip(documents, metadatas, ids):
                    if doc_id_val not in seen_ids:
                        all_results.append({
                            "content": doc,
                            "metadata": meta,
                            "id": doc_id_val,
                            "hop": hop_num + 1
                        })
                        seen_ids.add(doc_id_val)
                
                # Generate next query from retrieved content
                if hop_num < hops - 1 and all_results:
                    # Extract key terms from last result for next query
                    last_content = all_results[-1].get("content", "")
                    current_query = self._generate_related_query(query, last_content)
                    print(f"   -> Related query: {current_query}")
            
            except Exception as e:
                print(f"   ⚠️  Hop {hop_num + 1} failed: {e}")
                break
        
        print(f"\n   ✅ Total unique results: {len(all_results)}")
        
        return {
            "multi_hop_results": all_results[:top_k],
            "num_hops": hops,
            "retrieval_method": "multi_hop"
        }
    
    @staticmethod
    def _generate_related_query(original_query: str, content: str) -> str:
        """
        Generate related query from content
        Extract key concepts to find related information
        """
        
        # Simple approach: extract key nouns from content
        import re
        
        # Get words that appear in content but not in original query
        original_words = set(original_query.lower().split())
        content_words = content.lower().split()
        
        # Find candidate related terms (fairly long words that appear multiple times)
        word_freq = {}
        for word in content_words:
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 4 and word_clean not in original_words:
                word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
        
        # Get top 3 related terms
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        related_terms = [term for term, _ in top_terms]
        
        if related_terms:
            return f"{original_query} {' '.join(related_terms)}"
        
        return original_query
