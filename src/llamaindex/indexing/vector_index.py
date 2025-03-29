"""
Vector Index implementation for LlamaIndex.

This module provides an implementation of a vector index for efficient semantic search
using embeddings.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime
import time

from llamaindex.core.index import Index
from llamaindex.core.document import Document, DocumentChunk
from llamaindex.core.query import Query, QueryResult, DocumentMatch


class VectorIndex(Index):
    """
    A vector index for semantic search with embeddings.
    
    This index stores document embeddings for fast similarity search based on
    the semantic meaning of queries and documents.
    """
    
    def __init__(
        self,
        index_id: Optional[str] = None,
        index_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        similarity_function: str = "cosine",
    ):
        """
        Initialize a vector index.
        
        Args:
            index_id: Optional unique identifier for the index
            index_name: Optional name for the index
            metadata: Optional metadata for the index
            embedding_function: Function that takes text and returns embedding vector 
                (if None, embeddings must be precomputed)
            similarity_function: Function to use for similarity calculations (default "cosine")
                Options: "cosine", "dot", "euclidean"
        """
        super().__init__(index_id, index_name, metadata)
        
        # Embedding function
        self.embedding_function = embedding_function
        
        # Similarity function
        self.similarity_function = similarity_function
        
        # Main index structures
        self.document_embeddings: Dict[str, List[float]] = {}
        self.chunk_embeddings: Dict[str, List[float]] = {}
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        self.chunk_map: Dict[str, DocumentChunk] = {}
        
        # Set of document/chunk IDs that need embeddings
        self.embedding_queue: List[Tuple[str, str, bool]] = []  # (id, text, is_chunk)
    
    def add_document(self, document: Document) -> None:
        """
        Add a document to the index.
        
        Args:
            document: The document to add
        """
        if document.document_id in self.documents:
            # Update the document
            self.delete_document(document.document_id)
        
        self.documents[document.document_id] = document
        self.document_ids.add(document.document_id)
        
        # If document has no chunks, embed the full document
        if not document.chunks:
            if hasattr(document, 'embedding') and document.embedding is not None:
                self.document_embeddings[document.document_id] = document.embedding
            elif self.embedding_function:
                embedding = self.embedding_function(document.text)
                self.document_embeddings[document.document_id] = embedding
            else:
                # Queue for later embedding
                self.embedding_queue.append((document.document_id, document.text, False))
            
            # Update stats
            self._stats["num_documents"] += 1
            self._stats["num_chunks"] += 1
        else:
            # Process each chunk
            for chunk in document.chunks:
                self.chunk_map[chunk.chunk_id] = chunk
                
                if chunk.embedding is not None:
                    self.chunk_embeddings[chunk.chunk_id] = chunk.embedding
                elif self.embedding_function:
                    embedding = self.embedding_function(chunk.text)
                    self.chunk_embeddings[chunk.chunk_id] = embedding
                else:
                    # Queue for later embedding
                    self.embedding_queue.append((chunk.chunk_id, chunk.text, True))
            
            # Update stats
            self._stats["num_documents"] += 1
            self._stats["num_chunks"] += len(document.chunks)
        
        # Update index stats
        self._stats["last_indexed"] = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the index.
        
        Args:
            documents: The documents to add
        """
        for document in documents:
            self.add_document(document)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if the document was deleted, False if it was not found
        """
        if document_id not in self.documents:
            return False
        
        # Get the document
        document = self.documents[document_id]
        
        # Remove document embedding
        if document_id in self.document_embeddings:
            del self.document_embeddings[document_id]
        
        # Remove any chunks
        if document.chunks:
            for chunk in document.chunks:
                if chunk.chunk_id in self.chunk_embeddings:
                    del self.chunk_embeddings[chunk.chunk_id]
                if chunk.chunk_id in self.chunk_map:
                    del self.chunk_map[chunk.chunk_id]
        
        # Remove from documents
        del self.documents[document_id]
        self.document_ids.remove(document_id)
        
        # Update stats
        self._stats["num_documents"] -= 1
        self._stats["num_chunks"] -= len(document.chunks) if document.chunks else 1
        
        return True
    
    def clear(self) -> None:
        """Clear the index, removing all documents."""
        self.document_embeddings.clear()
        self.chunk_embeddings.clear()
        self.documents.clear()
        self.chunk_map.clear()
        self.document_ids.clear()
        self.embedding_queue.clear()
        
        # Reset stats
        self._stats = {
            "num_documents": 0,
            "num_chunks": 0,
            "num_tokens": 0,
            "num_queries": 0,
            "last_indexed": None,
            "last_queried": None,
        }
    
    def query(self, query: Query) -> QueryResult:
        """
        Execute a query against the index.
        
        Args:
            query: The query to execute
            
        Returns:
            QueryResult object containing matches and metadata
        """
        start_time = time.perf_counter()
        
        # If keyword search is required and this is just a vector index, warn but continue
        if query.search_type == "keyword":
            print("Warning: Keyword search requested but VectorIndex only supports semantic search.")
        
        # Get query embedding
        query_embedding = None
        
        if query.embedding is not None:
            query_embedding = query.embedding
        elif self.embedding_function:
            query_embedding = self.embedding_function(query.query_text)
        else:
            raise ValueError("No embedding function provided and query does not have an embedding")
        
        # Calculate similarity scores
        chunk_scores = {}
        doc_scores = {}
        
        # Get scores for chunks
        for chunk_id, embedding in self.chunk_embeddings.items():
            score = self._calculate_similarity(query_embedding, embedding)
            chunk_scores[chunk_id] = score
            
            # Also update the document score (maximum of chunk scores)
            chunk = self.chunk_map[chunk_id]
            doc_id = chunk.document_id
            
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score
        
        # Get scores for documents without chunks
        for doc_id, embedding in self.document_embeddings.items():
            score = self._calculate_similarity(query_embedding, embedding)
            doc_scores[doc_id] = score
        
        # Sort and get top_k results
        sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:query.top_k]
        
        # Prepare the matches
        matches = []
        for doc_id, score in sorted_scores:
            document = self.documents[doc_id]
            
            # If document has chunks, find the best matching chunk
            if document.chunks:
                best_chunk_id = None
                best_chunk_score = -float('inf')
                
                for chunk in document.chunks:
                    if chunk.chunk_id in chunk_scores:
                        chunk_score = chunk_scores[chunk.chunk_id]
                        if chunk_score > best_chunk_score:
                            best_chunk_score = chunk_score
                            best_chunk_id = chunk.chunk_id
                
                if best_chunk_id:
                    best_chunk = self.chunk_map[best_chunk_id]
                    
                    match = DocumentMatch(
                        document_id=doc_id,
                        chunk_id=best_chunk_id,
                        score=score,  # Use document score for consistency
                        text=best_chunk.text,
                        document=document,
                        chunk=best_chunk,
                        metadata=document.metadata.copy()
                    )
                else:
                    # Fallback to the full document if no chunks matched
                    match = DocumentMatch(
                        document_id=doc_id,
                        score=score,
                        text=document.text,
                        document=document,
                        metadata=document.metadata.copy()
                    )
            else:
                # For documents without chunks
                match = DocumentMatch(
                    document_id=doc_id,
                    score=score,
                    text=document.text,
                    document=document,
                    metadata=document.metadata.copy()
                )
            
            matches.append(match)
        
        # Calculate execution time
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Update stats
        self._stats["num_queries"] += 1
        self._stats["last_queried"] = datetime.utcnow().isoformat()
        
        # Create result
        result = QueryResult(
            query=query,
            matches=matches,
            total_found=len(doc_scores),
            execution_time_ms=execution_time_ms,
            metadata={
                "index_type": "VectorIndex",
                "similarity_function": self.similarity_function,
            }
        )
        
        return result
    
    def persist(self, directory: str) -> None:
        """
        Persist the index to disk.
        
        Args:
            directory: The directory to persist the index to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save embeddings
        with open(os.path.join(directory, "document_embeddings.pkl"), "wb") as f:
            pickle.dump(self.document_embeddings, f)
        
        with open(os.path.join(directory, "chunk_embeddings.pkl"), "wb") as f:
            pickle.dump(self.chunk_embeddings, f)
        
        # Save documents and chunks
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(directory, "chunk_map.pkl"), "wb") as f:
            pickle.dump(self.chunk_map, f)
        
        # Save metadata and settings
        metadata = {
            "index_id": self.index_id,
            "index_name": self.index_name,
            "index_type": self.__class__.__name__,
            "metadata": self.metadata,
            "document_ids": list(self.document_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "stats": self._stats,
            "similarity_function": self.similarity_function,
            "embedding_queue": [(id, text, is_chunk) for id, text, is_chunk in self.embedding_queue]
        }
        
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, directory: str, embedding_function: Optional[Callable[[str], List[float]]] = None) -> "VectorIndex":
        """
        Load an index from disk.
        
        Args:
            directory: The directory to load the index from
            embedding_function: Optional embedding function to use for new documents
            
        Returns:
            The loaded index
        """
        # Load metadata and settings
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Create index instance
        index = cls(
            index_id=metadata["index_id"],
            index_name=metadata["index_name"],
            metadata=metadata["metadata"],
            embedding_function=embedding_function,
            similarity_function=metadata["similarity_function"],
        )
        
        # Set timestamps
        index.created_at = datetime.fromisoformat(metadata["created_at"])
        index.updated_at = datetime.fromisoformat(metadata["updated_at"])
        
        # Set stats
        index._stats = metadata["stats"]
        
        # Load embeddings
        with open(os.path.join(directory, "document_embeddings.pkl"), "rb") as f:
            index.document_embeddings = pickle.load(f)
        
        with open(os.path.join(directory, "chunk_embeddings.pkl"), "rb") as f:
            index.chunk_embeddings = pickle.load(f)
        
        # Load documents and chunks
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            index.documents = pickle.load(f)
        
        with open(os.path.join(directory, "chunk_map.pkl"), "rb") as f:
            index.chunk_map = pickle.load(f)
        
        # Set document IDs
        index.document_ids = set(metadata["document_ids"])
        
        # Set embedding queue
        if "embedding_queue" in metadata:
            index.embedding_queue = metadata["embedding_queue"]
        
        return index
    
    def process_embedding_queue(self) -> int:
        """
        Process any documents or chunks waiting for embeddings.
        
        This requires an embedding function to be set.
        
        Returns:
            Number of items processed
        """
        if not self.embedding_function:
            raise ValueError("No embedding function provided")
        
        processed = 0
        
        while self.embedding_queue:
            id, text, is_chunk = self.embedding_queue.pop(0)
            embedding = self.embedding_function(text)
            
            if is_chunk:
                self.chunk_embeddings[id] = embedding
            else:
                self.document_embeddings[id] = embedding
            
            processed += 1
        
        return processed
    
    def set_embedding_function(self, embedding_function: Callable[[str], List[float]]) -> None:
        """
        Set the embedding function for the index.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
        """
        self.embedding_function = embedding_function
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score
        """
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        if self.similarity_function == "cosine":
            # Cosine similarity
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(v1, v2) / (norm1 * norm2)
        
        elif self.similarity_function == "dot":
            # Dot product
            return np.dot(v1, v2)
        
        elif self.similarity_function == "euclidean":
            # Euclidean distance (converted to similarity)
            dist = np.linalg.norm(v1 - v2)
            return 1.0 / (1.0 + dist)
        
        else:
            raise ValueError(f"Unsupported similarity function: {self.similarity_function}") 