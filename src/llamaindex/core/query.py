"""
Query classes for LlamaIndex.

This module provides classes for representing queries, query results, and query matches.
"""

import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class QueryMatch:
    """
    Represents a match from a search query.
    
    A query match contains information about a document or chunk that matched
    a search query, including the content, metadata, and score.
    """
    
    def __init__(
        self,
        doc_id: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ):
        """
        Initialize a QueryMatch.
        
        Args:
            doc_id: ID of the document that matched
            content: Text content of the match
            score: Relevance score for the match
            metadata: Metadata associated with the document/chunk
            chunk_id: ID of the chunk if the match is a document chunk
            embedding: Vector embedding of the match
        """
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.embedding = embedding
    
    def __str__(self) -> str:
        if self.chunk_id:
            return f"Match(doc='{self.doc_id}', chunk='{self.chunk_id}', score={self.score:.3f})"
        return f"Match(doc='{self.doc_id}', score={self.score:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the match to a dictionary.
        
        Returns:
            Dictionary representation of the match
        """
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryMatch':
        """
        Create a QueryMatch from a dictionary.
        
        Args:
            data: Dictionary representation of a match
            
        Returns:
            QueryMatch object
        """
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            score=data["score"],
            metadata=data.get("metadata", {}),
            chunk_id=data.get("chunk_id"),
            embedding=data.get("embedding"),
        )


class Query:
    """
    Represents a search query.
    
    A query encapsulates all the parameters needed to execute a search against an index,
    including the query text, filters, and other configuration options.
    """
    
    def __init__(
        self,
        query_text: str,
        query_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        search_type: str = "hybrid",
    ):
        """
        Initialize a Query.
        
        Args:
            query_text: The text of the query
            query_id: Unique identifier for the query
            top_k: Number of results to return
            filters: Filters to apply to the query
            embedding: Vector embedding of the query text
            search_type: Type of search to perform
                Options: "keyword", "semantic", "hybrid"
        """
        self.query_text = query_text
        self.query_id = query_id or str(uuid.uuid4())
        self.top_k = top_k
        self.filters = filters or {}
        self.embedding = embedding
        self.search_type = search_type
        self.created_at = datetime.utcnow()
    
    def __str__(self) -> str:
        return f"Query('{self.query_text}', type={self.search_type})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the query to a dictionary.
        
        Returns:
            Dictionary representation of the query
        """
        return {
            "query_text": self.query_text,
            "query_id": self.query_id,
            "top_k": self.top_k,
            "filters": self.filters,
            "embedding": self.embedding,
            "search_type": self.search_type,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """
        Create a Query from a dictionary.
        
        Args:
            data: Dictionary representation of a query
            
        Returns:
            Query object
        """
        query = cls(
            query_text=data["query_text"],
            query_id=data.get("query_id"),
            top_k=data.get("top_k", 10),
            filters=data.get("filters", {}),
            embedding=data.get("embedding"),
            search_type=data.get("search_type", "hybrid"),
        )
        
        if "created_at" in data:
            query.created_at = datetime.fromisoformat(data["created_at"])
        
        return query


class QueryResult:
    """
    Represents the result of a search query.
    
    A query result contains the matches found for a given query, along with
    metadata about the search process and results.
    """
    
    def __init__(
        self,
        query_text: str,
        query_id: Optional[str] = None,
        search_type: str = "hybrid",
        matches: Optional[List[QueryMatch]] = None,
    ):
        """
        Initialize a QueryResult.
        
        Args:
            query_text: The text of the query
            query_id: Unique identifier for the query
            search_type: Type of search that was performed
            matches: List of matches found
        """
        self.query_text = query_text
        self.query_id = query_id or str(uuid.uuid4())
        self.search_type = search_type
        self.matches = matches or []
        self.created_at = datetime.utcnow()
        
        # Search metadata
        self.total_found = len(self.matches)
        self.execution_time_ms = 0.0
    
    def __str__(self) -> str:
        return f"QueryResult('{self.query_text[:20]}...', matches={len(self.matches)})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def add_match(self, match: QueryMatch) -> None:
        """
        Add a match to the result.
        
        Args:
            match: QueryMatch to add
        """
        self.matches.append(match)
        self.total_found = len(self.matches)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "query_text": self.query_text,
            "query_id": self.query_id,
            "search_type": self.search_type,
            "matches": [match.to_dict() for match in self.matches],
            "total_found": self.total_found,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """
        Create a QueryResult from a dictionary.
        
        Args:
            data: Dictionary representation of a result
            
        Returns:
            QueryResult object
        """
        result = cls(
            query_text=data["query_text"],
            query_id=data.get("query_id"),
            search_type=data.get("search_type", "hybrid"),
        )
        
        if "matches" in data:
            result.matches = [QueryMatch.from_dict(match_data) for match_data in data["matches"]]
            result.total_found = len(result.matches)
        
        if "execution_time_ms" in data:
            result.execution_time_ms = data["execution_time_ms"]
        
        if "created_at" in data:
            result.created_at = datetime.fromisoformat(data["created_at"])
        
        return result 