"""
Search strategies for LlamaIndex.

This module provides search strategy implementations that can be used with the
QueryProcessor to execute different types of searches.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime

from llamaindex.core.index import Index
from llamaindex.core.query import Query, QueryResult
from llamaindex.indexing.inverted_index import InvertedIndex
from llamaindex.indexing.vector_index import VectorIndex
from llamaindex.indexing.hybrid_index import HybridIndex


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.
    
    A search strategy defines how to execute a particular type of search
    against an index or multiple indexes.
    """
    
    @abstractmethod
    def search(
        self,
        query: Query,
        index: Optional[Index] = None,
        indices: Optional[Dict[str, Index]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Execute a search against one or more indexes.
        
        Args:
            query: Query to execute
            index: Optional single index to search
            indices: Optional dictionary of indexes to search
            **kwargs: Additional arguments for the search
            
        Returns:
            QueryResult containing matches and metadata
        """
        pass


class BM25Search(SearchStrategy):
    """
    BM25-based keyword search strategy.
    
    This strategy performs keyword search using an inverted index and the
    BM25 ranking algorithm.
    """
    
    def __init__(
        self,
        use_idf: bool = True,
        minimum_should_match: float = 0.0,
    ):
        """
        Initialize a BM25Search strategy.
        
        Args:
            use_idf: Whether to use inverse document frequency in scoring
            minimum_should_match: Minimum fraction of query terms that should match
        """
        self.use_idf = use_idf
        self.minimum_should_match = minimum_should_match
    
    def search(
        self,
        query: Query,
        index: Optional[Index] = None,
        indices: Optional[Dict[str, Index]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Execute a keyword search using BM25 ranking.
        
        Args:
            query: Query to execute
            index: Optional single index to search
            indices: Optional dictionary of indexes to search
            **kwargs: Additional arguments for the search
            
        Returns:
            QueryResult containing matches and metadata
        """
        # Make sure we have an index
        if index is None and (indices is None or not indices):
            raise ValueError("Either index or indices must be provided")
        
        # Force search type to keyword
        keyword_query = Query(
            query_text=query.query_text,
            filters=query.filters,
            top_k=query.top_k,
            search_type="keyword",
            query_id=query.query_id,
        )
        
        # If we have a single index
        if index is not None:
            if isinstance(index, InvertedIndex):
                # Set index parameters if they differ
                original_use_idf = index.use_idf
                original_msm = index.minimum_should_match
                
                if self.use_idf != index.use_idf:
                    index.use_idf = self.use_idf
                
                if self.minimum_should_match != index.minimum_should_match:
                    index.minimum_should_match = self.minimum_should_match
                
                # Execute query
                result = index.query(keyword_query)
                
                # Restore original parameters
                if self.use_idf != original_use_idf:
                    index.use_idf = original_use_idf
                
                if self.minimum_should_match != original_msm:
                    index.minimum_should_match = original_msm
                
                return result
            
            elif isinstance(index, HybridIndex):
                # Use the inverted index component of the hybrid index
                return index.query(keyword_query)
            
            else:
                raise ValueError(f"Index type {type(index)} does not support keyword search")
        
        # If we have multiple indexes
        else:
            assert indices is not None, "indices must be provided"
            
            # Find an inverted index to use
            for idx_name, idx in indices.items():
                if isinstance(idx, InvertedIndex) or isinstance(idx, HybridIndex):
                    return self.search(query, index=idx)
            
            raise ValueError("No suitable index found for keyword search")


class VectorSearch(SearchStrategy):
    """
    Vector-based semantic search strategy.
    
    This strategy performs semantic search using a vector index and computes
    similarity between query and document embeddings.
    """
    
    def __init__(
        self,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        similarity_function: str = "cosine",
    ):
        """
        Initialize a VectorSearch strategy.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
            similarity_function: Similarity function to use
                Options: "cosine", "dot_product", "euclidean"
        """
        self.embedding_function = embedding_function
        self.similarity_function = similarity_function
    
    def search(
        self,
        query: Query,
        index: Optional[Index] = None,
        indices: Optional[Dict[str, Index]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Execute a semantic search using vector similarity.
        
        Args:
            query: Query to execute
            index: Optional single index to search
            indices: Optional dictionary of indexes to search
            **kwargs: Additional arguments for the search
            
        Returns:
            QueryResult containing matches and metadata
        """
        # Make sure we have an index
        if index is None and (indices is None or not indices):
            raise ValueError("Either index or indices must be provided")
        
        # Force search type to semantic
        semantic_query = Query(
            query_text=query.query_text,
            filters=query.filters,
            top_k=query.top_k,
            search_type="semantic",
            query_id=query.query_id,
            embedding=query.embedding,
        )
        
        # If query doesn't have embedding but we have embedding function, compute it
        if semantic_query.embedding is None and self.embedding_function is not None:
            semantic_query.embedding = self.embedding_function(semantic_query.query_text)
        
        # If we have a single index
        if index is not None:
            if isinstance(index, VectorIndex):
                # Set similarity function if it differs
                original_sim_fn = index.similarity_function
                
                if self.similarity_function != index.similarity_function:
                    index.similarity_function = self.similarity_function
                
                # Set embedding function if provided
                if self.embedding_function is not None:
                    index.set_embedding_function(self.embedding_function)
                
                # Execute query
                result = index.query(semantic_query)
                
                # Restore original similarity function
                if self.similarity_function != original_sim_fn:
                    index.similarity_function = original_sim_fn
                
                return result
            
            elif isinstance(index, HybridIndex):
                # Use the vector index component of the hybrid index
                return index.query(semantic_query)
            
            else:
                raise ValueError(f"Index type {type(index)} does not support semantic search")
        
        # If we have multiple indexes
        else:
            assert indices is not None, "indices must be provided"
            
            # Find a vector index to use
            for idx_name, idx in indices.items():
                if isinstance(idx, VectorIndex) or isinstance(idx, HybridIndex):
                    return self.search(query, index=idx)
            
            raise ValueError("No suitable index found for semantic search")


class HybridSearch(SearchStrategy):
    """
    Hybrid search strategy combining keyword and semantic search.
    
    This strategy performs both keyword (BM25) and semantic (vector) searches
    and combines the results with a weighted score.
    """
    
    def __init__(
        self,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        alpha: float = 0.5,
        minimum_should_match: float = 0.0,
        similarity_function: str = "cosine",
    ):
        """
        Initialize a HybridSearch strategy.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
            alpha: Weight to give to semantic search results (0.0-1.0)
                  0.0 = 100% keyword search, 1.0 = 100% semantic search
            minimum_should_match: Minimum fraction of query terms that should match
            similarity_function: Similarity function to use for vector search
                Options: "cosine", "dot_product", "euclidean"
        """
        self.alpha = max(0.0, min(1.0, alpha))  # Ensure alpha is between 0 and 1
        self.embedding_function = embedding_function
        self.minimum_should_match = minimum_should_match
        self.similarity_function = similarity_function
        
        # Create sub-strategies
        self.keyword_search = BM25Search(
            use_idf=True,
            minimum_should_match=minimum_should_match,
        )
        
        self.semantic_search = VectorSearch(
            embedding_function=embedding_function,
            similarity_function=similarity_function,
        )
    
    def search(
        self,
        query: Query,
        index: Optional[Index] = None,
        indices: Optional[Dict[str, Index]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Execute a hybrid search combining keyword and semantic results.
        
        Args:
            query: Query to execute
            index: Optional single index to search
            indices: Optional dictionary of indexes to search
            **kwargs: Additional arguments for the search
            
        Returns:
            QueryResult containing matches and metadata
        """
        # Make sure we have an index
        if index is None and (indices is None or not indices):
            raise ValueError("Either index or indices must be provided")
        
        # If we have a specialized HybridIndex, use it directly
        if index is not None and isinstance(index, HybridIndex):
            # Set alpha if it differs
            original_alpha = index.hybrid_alpha
            
            if self.alpha != index.hybrid_alpha:
                index.hybrid_alpha = self.alpha
            
            # Set embedding function if provided
            if self.embedding_function is not None:
                index.set_embedding_function(self.embedding_function)
            
            # Create a hybrid query
            hybrid_query = Query(
                query_text=query.query_text,
                filters=query.filters,
                top_k=query.top_k,
                search_type="hybrid",
                query_id=query.query_id,
                embedding=query.embedding,
            )
            
            # Execute query
            result = index.query(hybrid_query)
            
            # Restore original alpha
            if self.alpha != original_alpha:
                index.hybrid_alpha = original_alpha
            
            return result
        
        # For other index types, we need to manually combine results
        
        # Get keyword and semantic search results
        keyword_results = None
        semantic_results = None
        
        # Skip keyword search if alpha is 1.0 (pure semantic)
        if self.alpha < 1.0:
            try:
                keyword_results = self.keyword_search.search(query, index, indices)
            except ValueError:
                # If keyword search fails, adjust alpha to use only semantic
                if self.alpha < 1.0:
                    self.alpha = 1.0
        
        # Skip semantic search if alpha is 0.0 (pure keyword)
        if self.alpha > 0.0:
            try:
                semantic_results = self.semantic_search.search(query, index, indices)
            except ValueError:
                # If semantic search fails, adjust alpha to use only keyword
                if self.alpha > 0.0:
                    self.alpha = 0.0
        
        # If both searches failed, we have a problem
        if keyword_results is None and semantic_results is None:
            raise ValueError("Neither keyword nor semantic search could be performed")
        
        # If only one search succeeded, return those results
        if keyword_results is None:
            return semantic_results
        
        if semantic_results is None:
            return keyword_results
        
        # Combine results
        return self._combine_results(keyword_results, semantic_results, query.top_k)
    
    def _combine_results(
        self, 
        keyword_results: QueryResult, 
        semantic_results: QueryResult, 
        top_k: int
    ) -> QueryResult:
        """
        Combine keyword and semantic search results with weighted scoring.
        
        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            top_k: Number of top results to return
            
        Returns:
            Combined QueryResult
        """
        # Create a dictionary to store combined scores
        combined_scores: Dict[str, Dict] = {}
        
        # Process keyword results
        if keyword_results and keyword_results.matches:
            keyword_weight = 1.0 - self.alpha
            for match in keyword_results.matches:
                doc_id = match.doc_id
                chunk_id = match.chunk_id if hasattr(match, 'chunk_id') else None
                key = f"{doc_id}:{chunk_id}" if chunk_id else doc_id
                
                combined_scores[key] = {
                    "match": match,
                    "score": match.score * keyword_weight,
                    "sources": ["keyword"],
                }
        
        # Process semantic results
        if semantic_results and semantic_results.matches:
            semantic_weight = self.alpha
            for match in semantic_results.matches:
                doc_id = match.doc_id
                chunk_id = match.chunk_id if hasattr(match, 'chunk_id') else None
                key = f"{doc_id}:{chunk_id}" if chunk_id else doc_id
                
                if key in combined_scores:
                    # Add to existing entry
                    combined_scores[key]["score"] += match.score * semantic_weight
                    combined_scores[key]["sources"].append("semantic")
                else:
                    # Create new entry
                    combined_scores[key] = {
                        "match": match,
                        "score": match.score * semantic_weight,
                        "sources": ["semantic"],
                    }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        
        # Take top_k results
        top_results = sorted_results[:top_k]
        
        # Create a new result with combined matches
        result = QueryResult(
            query_text=keyword_results.query_text,
            query_id=keyword_results.query_id,
            search_type="hybrid",
        )
        
        # Update matches with combined scores
        for item in top_results:
            match = item["match"]
            match.score = item["score"]  # Update score to combined score
            result.add_match(match)
        
        return result 