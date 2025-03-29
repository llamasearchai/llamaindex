"""
Base embeddings interface for LlamaIndex
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

class Embeddings(ABC):
    """
    Base class for embeddings in LlamaIndex.
    
    Embeddings transform text into vector representations that can be used
    for semantic search and other NLP tasks.
    """
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Return the dimensionality of the embeddings.
        
        Returns:
            The number of dimensions in the embeddings
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embeddings, one per document
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text.
        
        Some embedding models use different embeddings for queries vs documents.
        
        Args:
            text: Query text to embed
            
        Returns:
            The query embedding
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert embeddings to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the embeddings
        """
        return {
            "type": self.__class__.__name__,
            "dimensions": self.dimensions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Embeddings":
        """
        Create embeddings from a dictionary.
        
        Args:
            data: Dictionary representation of embeddings
            
        Returns:
            Embeddings instance
        """
        # This should be implemented by subclasses
        raise NotImplementedError(
            f"from_dict is not implemented for {cls.__name__}"
        ) 