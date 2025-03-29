"""
Index interface for LlamaIndex.

This module provides the base Index class that all index implementations must extend.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set
import os
import json
from datetime import datetime

from llamaindex.core.document import Document
from llamaindex.core.query import Query, QueryResult


class Index(ABC):
    """
    Abstract base class for all index implementations.
    
    An index is responsible for storing, indexing, and retrieving documents
    based on queries. Different index implementations use different strategies
    for indexing and retrieval.
    """
    
    def __init__(
        self,
        index_id: Optional[str] = None,
        index_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an Index.
        
        Args:
            index_id: Unique identifier for the index
            index_name: Human-readable name for the index
            metadata: Additional metadata for the index
        """
        import uuid
        
        self.index_id = index_id or str(uuid.uuid4())
        self.index_name = index_name or f"index-{self.index_id[:8]}"
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.last_updated = self.created_at
        
        # Track document IDs in the index
        self.doc_ids: Set[str] = set()
    
    @abstractmethod
    def add_document(self, document: Document) -> None:
        """
        Add a document to the index.
        
        Args:
            document: The document to add
        """
        pass
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the index.
        
        Args:
            documents: List of documents to add
        """
        for document in documents:
            self.add_document(document)
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            True if the document was deleted, False if it wasn't found
        """
        pass
    
    @abstractmethod
    def query(self, query: Query) -> QueryResult:
        """
        Execute a query against the index.
        
        Args:
            query: The query to execute
            
        Returns:
            QueryResult containing matches and metadata
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the index."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "index_id": self.index_id,
            "index_name": self.index_name,
            "doc_count": len(self.doc_ids),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }
    
    def persist(self, directory: str) -> str:
        """
        Persist the index to disk.
        
        This base implementation saves the index metadata. Subclasses should
        override this to save their specific index data, and call super().persist()
        
        Args:
            directory: Directory to save the index to
            
        Returns:
            Path to the saved index
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(directory, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "index_id": self.index_id,
                "index_name": self.index_name,
                "index_type": self.__class__.__name__,
                "metadata": self.metadata,
                "doc_ids": list(self.doc_ids),
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
            }, f, indent=2)
        
        return directory
    
    @classmethod
    def load(cls, directory: str) -> 'Index':
        """
        Load an index from disk.
        
        This is a stub that should be implemented by subclasses.
        
        Args:
            directory: Directory where the index is saved
            
        Returns:
            Loaded index
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def _update_metadata(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.utcnow() 