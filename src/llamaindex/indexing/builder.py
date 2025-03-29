"""
Index Builder for LlamaIndex.

This module provides utilities for building and configuring indexes.
"""

import os
from typing import Dict, List, Optional, Any, Callable, Union, Type, Set
from datetime import datetime

from llamaindex.core.index import Index
from llamaindex.core.document import Document, DocumentChunk
from llamaindex.indexing.inverted_index import InvertedIndex
from llamaindex.indexing.vector_index import VectorIndex
from llamaindex.indexing.hybrid_index import HybridIndex


class IndexBuilder:
    """
    A utility class for building and configuring indexes.
    
    This class provides a fluent interface for creating and configuring
    different types of indexes, with sensible defaults.
    """
    
    def __init__(self):
        """Initialize an IndexBuilder."""
        self.index_type: Optional[Type[Index]] = None
        self.index_id: Optional[str] = None
        self.index_name: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.embedding_function: Optional[Callable[[str], List[float]]] = None
        
        # InvertedIndex settings
        self.use_idf: bool = True
        self.minimum_should_match: float = 0.0
        
        # VectorIndex settings
        self.similarity_function: str = "cosine"
        
        # HybridIndex settings
        self.hybrid_alpha: float = 0.5
        
        # Documents to add on build
        self.documents: List[Document] = []
    
    def with_index_type(self, index_type: Union[str, Type[Index]]) -> "IndexBuilder":
        """
        Set the type of index to build.
        
        Args:
            index_type: Either a string ("inverted", "vector", "hybrid") or an Index class
            
        Returns:
            The builder instance for chaining
        """
        if isinstance(index_type, str):
            index_type_str = index_type.lower()
            if index_type_str == "inverted":
                self.index_type = InvertedIndex
            elif index_type_str == "vector":
                self.index_type = VectorIndex
            elif index_type_str == "hybrid":
                self.index_type = HybridIndex
            else:
                raise ValueError(f"Unknown index type: {index_type}")
        else:
            self.index_type = index_type
        
        return self
    
    def with_id(self, index_id: str) -> "IndexBuilder":
        """
        Set the ID of the index.
        
        Args:
            index_id: The index ID
            
        Returns:
            The builder instance for chaining
        """
        self.index_id = index_id
        return self
    
    def with_name(self, index_name: str) -> "IndexBuilder":
        """
        Set the name of the index.
        
        Args:
            index_name: The index name
            
        Returns:
            The builder instance for chaining
        """
        self.index_name = index_name
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> "IndexBuilder":
        """
        Set the metadata for the index.
        
        Args:
            metadata: The index metadata
            
        Returns:
            The builder instance for chaining
        """
        self.metadata = metadata
        return self
    
    def with_embedding_function(self, embedding_function: Callable[[str], List[float]]) -> "IndexBuilder":
        """
        Set the embedding function for the index.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
            
        Returns:
            The builder instance for chaining
        """
        self.embedding_function = embedding_function
        return self
    
    def with_idf(self, use_idf: bool) -> "IndexBuilder":
        """
        Set whether to use inverse document frequency in scoring for inverted indexes.
        
        Args:
            use_idf: Whether to use IDF
            
        Returns:
            The builder instance for chaining
        """
        self.use_idf = use_idf
        return self
    
    def with_minimum_should_match(self, minimum_should_match: float) -> "IndexBuilder":
        """
        Set the minimum percentage of query terms that should match for inverted indexes.
        
        Args:
            minimum_should_match: Minimum percentage (0.0-1.0)
            
        Returns:
            The builder instance for chaining
        """
        self.minimum_should_match = minimum_should_match
        return self
    
    def with_similarity_function(self, similarity_function: str) -> "IndexBuilder":
        """
        Set the similarity function for vector indexes.
        
        Args:
            similarity_function: Similarity function ("cosine", "dot", "euclidean")
            
        Returns:
            The builder instance for chaining
        """
        self.similarity_function = similarity_function
        return self
    
    def with_hybrid_alpha(self, hybrid_alpha: float) -> "IndexBuilder":
        """
        Set the hybrid weighting parameter for hybrid indexes.
        
        Args:
            hybrid_alpha: Weight for combining scores (0.0-1.0)
            
        Returns:
            The builder instance for chaining
        """
        self.hybrid_alpha = hybrid_alpha
        return self
    
    def with_documents(self, documents: List[Document]) -> "IndexBuilder":
        """
        Set documents to add to the index on build.
        
        Args:
            documents: List of documents to add
            
        Returns:
            The builder instance for chaining
        """
        self.documents = documents
        return self
    
    def add_document(self, document: Document) -> "IndexBuilder":
        """
        Add a document to the list of documents to add on build.
        
        Args:
            document: Document to add
            
        Returns:
            The builder instance for chaining
        """
        self.documents.append(document)
        return self
    
    def from_existing(self, index: Index) -> "IndexBuilder":
        """
        Initialize the builder from an existing index.
        
        Args:
            index: Existing index to use as template
            
        Returns:
            The builder instance for chaining
        """
        self.index_type = index.__class__
        self.index_id = index.index_id
        self.index_name = index.index_name
        self.metadata = index.metadata.copy()
        
        if isinstance(index, VectorIndex) or isinstance(index, HybridIndex):
            if isinstance(index, VectorIndex):
                self.embedding_function = index.embedding_function
                self.similarity_function = index.similarity_function
            elif isinstance(index, HybridIndex):
                self.embedding_function = index.vector_index.embedding_function
                self.similarity_function = index.vector_index.similarity_function
                self.use_idf = index.keyword_index.use_idf
                self.minimum_should_match = index.keyword_index.minimum_should_match
                self.hybrid_alpha = index.hybrid_alpha
        elif isinstance(index, InvertedIndex):
            self.use_idf = index.use_idf
            self.minimum_should_match = index.minimum_should_match
        
        return self
    
    def build(self) -> Index:
        """
        Build the index based on the configured settings.
        
        Returns:
            The constructed index
        """
        if not self.index_type:
            # Default to hybrid index if not specified
            self.index_type = HybridIndex
        
        if self.index_type == InvertedIndex:
            index = InvertedIndex(
                index_id=self.index_id,
                index_name=self.index_name,
                metadata=self.metadata,
                use_idf=self.use_idf,
                minimum_should_match=self.minimum_should_match,
            )
        elif self.index_type == VectorIndex:
            index = VectorIndex(
                index_id=self.index_id,
                index_name=self.index_name,
                metadata=self.metadata,
                embedding_function=self.embedding_function,
                similarity_function=self.similarity_function,
            )
        elif self.index_type == HybridIndex:
            index = HybridIndex(
                index_id=self.index_id,
                index_name=self.index_name,
                metadata=self.metadata,
                embedding_function=self.embedding_function,
                use_idf=self.use_idf,
                minimum_should_match=self.minimum_should_match,
                similarity_function=self.similarity_function,
                hybrid_alpha=self.hybrid_alpha,
            )
        else:
            # Custom index type - pass all arguments and let it filter what it needs
            index = self.index_type(
                index_id=self.index_id,
                index_name=self.index_name,
                metadata=self.metadata,
                embedding_function=self.embedding_function,
                use_idf=self.use_idf,
                minimum_should_match=self.minimum_should_match,
                similarity_function=self.similarity_function,
                hybrid_alpha=self.hybrid_alpha,
            )
        
        # Add documents if provided
        if self.documents:
            index.add_documents(self.documents)
        
        return index
    
    @classmethod
    def load(cls, directory: str, embedding_function: Optional[Callable[[str], List[float]]] = None) -> Index:
        """
        Load an index from disk based on its metadata.
        
        Args:
            directory: The directory to load from
            embedding_function: Optional embedding function for vector indexes
            
        Returns:
            The loaded index
        """
        import json
        import os.path
        
        # Load metadata to determine index type
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        index_type = metadata.get("index_type")
        
        if index_type == "InvertedIndex":
            return InvertedIndex.load(directory)
        elif index_type == "VectorIndex":
            return VectorIndex.load(directory, embedding_function)
        elif index_type == "HybridIndex":
            return HybridIndex.load(directory, embedding_function)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    @classmethod
    def create_inverted_index(cls, index_name: Optional[str] = None) -> Index:
        """
        Create an inverted index with default settings.
        
        Args:
            index_name: Optional name for the index
            
        Returns:
            An inverted index
        """
        return cls().with_index_type(InvertedIndex).with_name(index_name).build()
    
    @classmethod
    def create_vector_index(
        cls, 
        embedding_function: Callable[[str], List[float]], 
        index_name: Optional[str] = None
    ) -> Index:
        """
        Create a vector index with default settings.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
            index_name: Optional name for the index
            
        Returns:
            A vector index
        """
        return (
            cls()
            .with_index_type(VectorIndex)
            .with_name(index_name)
            .with_embedding_function(embedding_function)
            .build()
        )
    
    @classmethod
    def create_hybrid_index(
        cls, 
        embedding_function: Callable[[str], List[float]], 
        index_name: Optional[str] = None
    ) -> Index:
        """
        Create a hybrid index with default settings.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
            index_name: Optional name for the index
            
        Returns:
            A hybrid index
        """
        return (
            cls()
            .with_index_type(HybridIndex)
            .with_name(index_name)
            .with_embedding_function(embedding_function)
            .build()
        ) 