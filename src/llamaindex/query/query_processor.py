"""
Query Processor for LlamaIndex.

This module provides the QueryProcessor for executing and processing queries.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import time

from llamaindex.core.index import Index
from llamaindex.core.query import Query, QueryResult


class QueryProcessor:
    """
    A processor for executing queries against indexes.
    
    This class provides utilities for executing and enhancing queries, handling
    caching, statistics, and other aspects of query processing.
    """
    
    def __init__(
        self,
        index: Optional[Index] = None,
        indices: Optional[Dict[str, Index]] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        max_query_cache_size: int = 100,
    ):
        """
        Initialize a QueryProcessor.
        
        Args:
            index: A single index to query
            indices: Multiple indexes to query, with string keys for identification
            embedding_function: Optional function to generate embeddings for queries
            max_query_cache_size: Maximum number of cached queries to keep
        """
        # Set up index(es)
        self.index = index
        self.indices = indices or {}
        if index:
            self.indices["default"] = index
        
        # Set up embedding function
        self.embedding_function = embedding_function
        
        # Query statistics
        self.query_count = 0
        self.total_query_time_ms = 0
        self.avg_query_time_ms = 0
        self.last_query_time = None
        
        # Query cache
        self.max_query_cache_size = max_query_cache_size
        self.query_cache: Dict[str, QueryResult] = {}
        self.query_cache_hits = 0
    
    def query(
        self,
        query_text: str,
        index_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        use_cache: bool = True,
    ) -> QueryResult:
        """
        Execute a query against an index.
        
        Args:
            query_text: The text of the query
            index_id: ID of the index to query (if multiple are available)
            top_k: Number of results to return
            filters: Optional filters to apply to the query
            search_type: Type of search to perform (default "hybrid")
                Options: "keyword", "semantic", "hybrid"
            use_cache: Whether to use query caching
            
        Returns:
            QueryResult object containing matches and metadata
        """
        # Get the index to query
        target_index = self._get_index(index_id)
        if not target_index:
            raise ValueError(f"No index found. Specify a valid index_id from: {list(self.indices.keys())}")
        
        # Generate cache key (if using cache)
        cache_key = None
        if use_cache:
            cache_components = [
                query_text,
                index_id or "default",
                top_k,
                str(filters) if filters else "",
                search_type,
            ]
            cache_key = "|".join([str(c) for c in cache_components])
            
            # Check cache
            if cache_key in self.query_cache:
                self.query_cache_hits += 1
                return self.query_cache[cache_key]
        
        # Prepare Query object
        query = Query(
            query_text=query_text,
            top_k=top_k,
            filters=filters,
            search_type=search_type,
        )
        
        # Add embedding if available
        if self.embedding_function and search_type in ["semantic", "hybrid"]:
            query.embedding = self.embedding_function(query_text)
        
        # Execute query
        start_time = time.perf_counter()
        result = target_index.query(query)
        end_time = time.perf_counter()
        
        # Update stats
        query_time_ms = (end_time - start_time) * 1000
        self.query_count += 1
        self.total_query_time_ms += query_time_ms
        self.avg_query_time_ms = self.total_query_time_ms / self.query_count
        self.last_query_time = datetime.utcnow()
        
        # Update cache if enabled
        if use_cache and cache_key:
            # Limit cache size
            if len(self.query_cache) >= self.max_query_cache_size:
                # Simple LRU: remove first entry (oldest)
                if self.query_cache:
                    self.query_cache.pop(next(iter(self.query_cache)))
            
            self.query_cache[cache_key] = result
        
        return result
    
    def bulk_query(
        self,
        queries: List[str],
        index_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        use_cache: bool = True,
    ) -> List[QueryResult]:
        """
        Execute multiple queries against an index.
        
        Args:
            queries: List of query texts
            index_id: ID of the index to query (if multiple are available)
            top_k: Number of results to return per query
            filters: Optional filters to apply to the queries
            search_type: Type of search to perform (default "hybrid")
            use_cache: Whether to use query caching
            
        Returns:
            List of QueryResult objects
        """
        results = []
        for query_text in queries:
            result = self.query(
                query_text=query_text,
                index_id=index_id,
                top_k=top_k,
                filters=filters,
                search_type=search_type,
                use_cache=use_cache,
            )
            results.append(result)
        
        return results
    
    def cross_query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        use_cache: bool = True,
    ) -> Dict[str, QueryResult]:
        """
        Execute a query against all available indexes.
        
        Args:
            query_text: The text of the query
            top_k: Number of results to return per index
            filters: Optional filters to apply to the query
            search_type: Type of search to perform (default "hybrid")
            use_cache: Whether to use query caching
            
        Returns:
            Dictionary mapping index IDs to QueryResult objects
        """
        results = {}
        for index_id in self.indices:
            result = self.query(
                query_text=query_text,
                index_id=index_id,
                top_k=top_k,
                filters=filters,
                search_type=search_type,
                use_cache=use_cache,
            )
            results[index_id] = result
        
        return results
    
    def add_index(self, index: Index, index_id: str) -> None:
        """
        Add an index to the processor.
        
        Args:
            index: The index to add
            index_id: ID to associate with the index
        """
        self.indices[index_id] = index
        
        # If this is the first index, also set it as the default
        if not self.index and len(self.indices) == 1:
            self.index = index
    
    def set_default_index(self, index_id: str) -> None:
        """
        Set the default index for queries.
        
        Args:
            index_id: ID of the index to set as default
        """
        if index_id not in self.indices:
            raise ValueError(f"No index with ID {index_id} found")
        
        self.index = self.indices[index_id]
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        self.query_cache_hits = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about query processing.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "query_count": self.query_count,
            "avg_query_time_ms": self.avg_query_time_ms,
            "total_query_time_ms": self.total_query_time_ms,
            "last_query_time": self.last_query_time.isoformat() if self.last_query_time else None,
            "cache_size": len(self.query_cache),
            "cache_hits": self.query_cache_hits,
            "cache_hit_rate": self.query_cache_hits / self.query_count if self.query_count > 0 else 0.0,
        }
    
    def set_embedding_function(self, embedding_function: Callable[[str], List[float]]) -> None:
        """
        Set the embedding function for query processing.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
        """
        self.embedding_function = embedding_function
    
    def _get_index(self, index_id: Optional[str] = None) -> Optional[Index]:
        """
        Get the index to query based on the index_id.
        
        Args:
            index_id: ID of the index to query (if multiple are available)
            
        Returns:
            The index to query, or None if no matching index is found
        """
        if index_id:
            return self.indices.get(index_id)
        
        return self.index or next(iter(self.indices.values())) if self.indices else None 