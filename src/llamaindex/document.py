"""
Document module for LlamaIndex
"""
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json

class Document:
    """
    Document class representing a document to be indexed.
    
    A document is the basic unit of indexing and searching. It consists of
    text content and optional metadata.
    """
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        chunks: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ):
        """
        Initialize a Document.
        
        Args:
            text: The text content of the document
            metadata: Optional metadata dictionary
            id: Optional document ID. If not provided, a UUID will be generated.
            chunks: Optional pre-chunked text. If not provided, the document will be chunked during indexing.
            embedding: Optional pre-computed embedding vector.
        """
        self.text = text
        self.metadata = metadata or {}
        self.id = id or f"doc-{uuid.uuid4()}"
        self.chunks = chunks
        self.embedding = embedding
        
        # Add indexed_at timestamp if not provided in metadata
        if "indexed_at" not in self.metadata:
            self.metadata["indexed_at"] = datetime.utcnow().isoformat()
    
    def __repr__(self) -> str:
        """Representation of the document."""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Document(id={self.id}, text=\"{preview}\", metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "chunks": self.chunks,
            "embedding": self.embedding,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create a Document from a dictionary."""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            chunks=data.get("chunks"),
            embedding=data.get("embedding"),
        )
    
    def to_json(self) -> str:
        """Convert the document to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """Create a Document from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def chunk_text(self, chunk_size: int = 512, chunk_overlap: int = 128) -> List[str]:
        """
        Split document text into chunks.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not self.text:
            return []
            
        # Simple chunking by character count
        if len(self.text) <= chunk_size:
            return [self.text]
        
        chunks = []
        start = 0
        while start < len(self.text):
            end = min(start + chunk_size, len(self.text))
            
            # Don't chunk in the middle of a word if possible
            if end < len(self.text):
                # Try to find a space or newline to break at
                while end > start and self.text[end] not in " \n\t.,:;!?":
                    end -= 1
                if end == start:  # If no good breakpoint, use original end
                    end = min(start + chunk_size, len(self.text))
            
            chunks.append(self.text[start:end].strip())
            start = end - chunk_overlap
        
        self.chunks = chunks
        return chunks 