"""
Results module for LlamaIndex
"""
from typing import Dict, Any, List, Optional, Union, Iterator
from pydantic import BaseModel, Field

from llamaindex.document import Document

class SearchResult:
    """
    A single search result.
    
    A search result includes the document, score, and any highlights.
    """
    
    def __init__(
        self,
        document: Document,
        score: float,
        highlights: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize a SearchResult.
        
        Args:
            document: The matched document
            score: The relevance score (higher is better)
            highlights: Optional dictionary of field -> highlight snippets
        """
        self.document = document
        self.score = score
        self.highlights = highlights or {}
    
    def __repr__(self) -> str:
        """Representation of the search result."""
        return f"SearchResult(score={self.score:.4f}, document={self.document})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the search result to a dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "highlights": self.highlights,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create a SearchResult from a dictionary."""
        return cls(
            document=Document.from_dict(data["document"]),
            score=data["score"],
            highlights=data.get("highlights"),
        )

class SearchResults:
    """
    Collection of search results.
    
    SearchResults provides an iterable interface to search results and
    includes metadata about the search query and performance.
    """
    
    def __init__(
        self,
        results: List[SearchResult],
        query: str,
        total: Optional[int] = None,
        took_ms: Optional[float] = None,
        facets: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """
        Initialize SearchResults.
        
        Args:
            results: List of search results
            query: The original search query
            total: Optional total number of results (may be more than len(results))
            took_ms: Optional time taken to perform the search in milliseconds
            facets: Optional facet counts (field -> value -> count)
        """
        self.results = results
        self.query = query
        self.total = total if total is not None else len(results)
        self.took_ms = took_ms
        self.facets = facets or {}
    
    def __len__(self) -> int:
        """Number of results."""
        return len(self.results)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[SearchResult, List[SearchResult]]:
        """Get result(s) by index."""
        return self.results[idx]
    
    def __iter__(self) -> Iterator[SearchResult]:
        """Iterate over results."""
        return iter(self.results)
    
    def __repr__(self) -> str:
        """Representation of the search results."""
        return f"SearchResults(query='{self.query}', total={self.total}, results={len(self.results)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the search results to a dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "total": self.total,
            "took_ms": self.took_ms,
            "facets": self.facets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResults":
        """Create SearchResults from a dictionary."""
        return cls(
            results=[SearchResult.from_dict(r) for r in data["results"]],
            query=data["query"],
            total=data.get("total"),
            took_ms=data.get("took_ms"),
            facets=data.get("facets"),
        )
    
    def get_documents(self) -> List[Document]:
        """Get the list of documents from the results."""
        return [result.document for result in self.results]
    
    def get_texts(self) -> List[str]:
        """Get the list of document texts from the results."""
        return [result.document.text for result in self.results] 