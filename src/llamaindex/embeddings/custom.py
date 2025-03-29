"""
Custom embeddings for LlamaIndex
"""
from typing import List, Dict, Any, Union, Optional, Callable
import logging

from llamaindex.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class CustomEmbeddings(Embeddings):
    """
    Custom embeddings for LlamaIndex.
    
    This class allows using a custom embedding function with LlamaIndex.
    """
    
    def __init__(
        self,
        embed_function: Callable[[List[str]], List[List[float]]],
        dimensions: int,
        query_function: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize CustomEmbeddings.
        
        Args:
            embed_function: Function that takes a list of texts and returns a list of embeddings
            dimensions: Dimensionality of the embeddings
            query_function: Optional separate function for embedding queries
        """
        self._embed_function = embed_function
        self._dimensions = dimensions
        self._query_function = query_function
    
    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self._dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embeddings, one per document
        """
        if not texts:
            return []
        
        return self._embed_function(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            The query embedding
        """
        if self._query_function is not None:
            return self._query_function(text)
        
        return self.embed_documents([text])[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert embeddings to a dictionary for serialization.
        
        Note: This method just stores metadata, not the actual functions.
        
        Returns:
            Dictionary representation of the embeddings
        """
        return {
            "type": self.__class__.__name__,
            "dimensions": self._dimensions,
            "has_query_function": self._query_function is not None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomEmbeddings":
        """
        Create CustomEmbeddings from a dictionary.
        
        This method is only a stub, as custom function references cannot be serialized.
        
        Args:
            data: Dictionary representation of embeddings
            
        Returns:
            CustomEmbeddings instance
        """
        raise NotImplementedError(
            "CustomEmbeddings cannot be deserialized without function references. "
            "Create a new instance with the appropriate functions."
        ) 