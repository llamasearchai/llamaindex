"""
Document classes for LlamaIndex.

This module provides the Document and DocumentChunk classes for representing
documents and document chunks in the search index.
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime


class DocumentChunk:
    """
    Represents a chunk of a document.
    
    A document chunk is a section of a document that can be indexed and
    searched independently. Chunks are typically created by splitting a
    document into smaller, more manageable pieces.
    """
    
    def __init__(
        self,
        content: str,
        chunk_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        position: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize a DocumentChunk.
        
        Args:
            content: Text content of the chunk
            chunk_id: Unique identifier for the chunk (auto-generated if not provided)
            doc_id: ID of the parent document
            metadata: Metadata associated with the chunk
            embedding: Vector embedding of the chunk content
            position: Tuple of (start, end) positions in the original document
        """
        self.content = content
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.doc_id = doc_id
        self.metadata = metadata or {}
        self.embedding = embedding
        self.position = position
        self.created_at = datetime.utcnow()
    
    def __str__(self) -> str:
        return f"Chunk(id='{self.chunk_id}', content='{self.content[:50]}...')"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the chunk to a dictionary.
        
        Returns:
            Dictionary representation of the chunk
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "position": self.position,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """
        Create a DocumentChunk from a dictionary.
        
        Args:
            data: Dictionary representation of a chunk
            
        Returns:
            DocumentChunk object
        """
        chunk = cls(
            content=data["content"],
            chunk_id=data.get("chunk_id"),
            doc_id=data.get("doc_id"),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            position=data.get("position"),
        )
        
        if "created_at" in data:
            chunk.created_at = datetime.fromisoformat(data["created_at"])
        
        return chunk


class Document:
    """
    Represents a document in the index.
    
    A document is a unit of content that can be indexed, retrieved, and
    searched. Documents can be chunked into smaller pieces for more efficient
    indexing and retrieval.
    """
    
    def __init__(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        chunks: Optional[List[DocumentChunk]] = None,
        source: Optional[str] = None,
    ):
        """
        Initialize a Document.
        
        Args:
            content: Text content of the document
            doc_id: Unique identifier for the document (auto-generated if not provided)
            metadata: Metadata associated with the document
            embedding: Vector embedding of the document content
            chunks: Pre-computed chunks of the document
            source: Source of the document (e.g., file path, URL)
        """
        self.content = content
        self.doc_id = doc_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.embedding = embedding
        self.chunks = chunks or []
        self.source = source
        self.created_at = datetime.utcnow()
        
        # Ensure chunks have the correct doc_id
        for chunk in self.chunks:
            chunk.doc_id = self.doc_id
    
    def __str__(self) -> str:
        return f"Document(id='{self.doc_id}', chunks={len(self.chunks)})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """
        Create a Document from a dictionary.
        
        Args:
            data: Dictionary representation of a document
            
        Returns:
            Document object
        """
        # First create document without chunks
        doc = cls(
            content=data["content"],
            doc_id=data.get("doc_id"),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            source=data.get("source"),
        )
        
        # Then add chunks if present
        if "chunks" in data:
            doc.chunks = [DocumentChunk.from_dict(chunk_data) for chunk_data in data["chunks"]]
            # Ensure all chunks have the correct doc_id
            for chunk in doc.chunks:
                chunk.doc_id = doc.doc_id
        
        if "created_at" in data:
            doc.created_at = datetime.fromisoformat(data["created_at"])
        
        return doc
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """
        Add a chunk to the document.
        
        Args:
            chunk: DocumentChunk to add
        """
        chunk.doc_id = self.doc_id
        self.chunks.append(chunk)
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add multiple chunks to the document.
        
        Args:
            chunks: List of DocumentChunk objects to add
        """
        for chunk in chunks:
            chunk.doc_id = self.doc_id
        self.chunks.extend(chunks)
    
    def clear_chunks(self) -> None:
        """Clear all chunks from the document."""
        self.chunks = []
    
    @property
    def num_chunks(self) -> int:
        """Get the number of chunks in the document."""
        return len(self.chunks)
    
    @property
    def text_length(self) -> int:
        """Get the length of the document content in characters."""
        return len(self.content) 