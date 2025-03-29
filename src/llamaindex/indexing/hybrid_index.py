"""
Hybrid Index implementation for LlamaIndex.

This module provides an implementation of a hybrid index that combines keyword-based
and vector-based search for high performance and relevance.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime
import time

from llamaindex.core.index import Index
from llamaindex.core.document import Document, DocumentChunk
from llamaindex.core.query import Query, QueryResult, DocumentMatch
from llamaindex.indexing.inverted_index import InvertedIndex
from llamaindex.indexing.vector_index import VectorIndex


class HybridIndex(Index):
    """
    A hybrid index that combines keyword and semantic search capabilities.
    
    This index maintains both an inverted index and a vector index internally,
    allowing for high-precision keyword search and high-recall semantic search
    to be combined.
    """
    
    def __init__(
        self,
        index_id: Optional[str] = None,
        index_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        use_idf: bool = True,
        minimum_should_match: float = 0.0,
        similarity_function: str = "cosine",
        hybrid_alpha: float = 0.5,
    ):
        """
        Initialize a hybrid index.
        
        Args:
            index_id: Optional unique identifier for the index
            index_name: Optional name for the index
            metadata: Optional metadata for the index
            embedding_function: Function that takes text and returns embedding vector
                (if None, vector index will require precomputed embeddings)
            use_idf: Whether to use inverse document frequency in scoring (default True)
            minimum_should_match: Minimum percentage of query terms that should match (0.0-1.0)
            similarity_function: Function to use for similarity calculations (default "cosine")
                Options: "cosine", "dot", "euclidean"
            hybrid_alpha: Weight for combining keyword and semantic scores (0.0-1.0)
                0.0 = pure keyword search, 1.0 = pure semantic search, 0.5 = balanced
        """
        super().__init__(index_id, index_name, metadata)
        
        # Create the component indexes
        self.keyword_index = InvertedIndex(
            index_id=f"{index_id}_keyword" if index_id else None,
            index_name=f"{index_name}_keyword" if index_name else None,
            use_idf=use_idf,
            minimum_should_match=minimum_should_match,
        )
        
        self.vector_index = VectorIndex(
            index_id=f"{index_id}_vector" if index_id else None,
            index_name=f"{index_name}_vector" if index_name else None,
            embedding_function=embedding_function,
            similarity_function=similarity_function,
        )
        
        # Set the hybrid weighting parameter
        self.hybrid_alpha = hybrid_alpha
        
        # Document storage
        self.documents: Dict[str, Document] = {}
    
    def add_document(self, document: Document) -> None:
        """
        Add a document to the index.
        
        Args:
            document: The document to add
        """
        # Store the document
        self.documents[document.document_id] = document
        self.document_ids.add(document.document_id)
        
        # Add to both sub-indexes
        self.keyword_index.add_document(document)
        self.vector_index.add_document(document)
        
        # Update stats (we only need to update ours, the sub-indexes track their own)
        self._stats["num_documents"] = len(self.document_ids)
        self._stats["num_chunks"] = self.keyword_index._stats["num_chunks"]
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
        
        # Delete from both sub-indexes
        keyword_result = self.keyword_index.delete_document(document_id)
        vector_result = self.vector_index.delete_document(document_id)
        
        # Delete from our document store
        del self.documents[document_id]
        self.document_ids.remove(document_id)
        
        # Update stats
        self._stats["num_documents"] = len(self.document_ids)
        self._stats["num_chunks"] = self.keyword_index._stats["num_chunks"]
        
        return keyword_result and vector_result
    
    def clear(self) -> None:
        """Clear the index, removing all documents."""
        self.keyword_index.clear()
        self.vector_index.clear()
        self.documents.clear()
        self.document_ids.clear()
        
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
        
        # Determine search strategy based on search_type
        if query.search_type == "keyword":
            # Keyword-only search
            return self.keyword_index.query(query)
        elif query.search_type == "semantic":
            # Semantic-only search
            return self.vector_index.query(query)
        
        # For hybrid search, query both indexes
        keyword_result = self.keyword_index.query(query)
        vector_result = self.vector_index.query(query)
        
        # Combine the results using the hybrid_alpha parameter
        combined_scores: Dict[str, Tuple[float, float, float, DocumentMatch]] = {}
        
        # Process keyword results
        for match in keyword_result.matches:
            combined_scores[match.document_id] = (
                match.score,  # keyword score
                0.0,  # vector score (will be updated if found)
                (1 - self.hybrid_alpha) * match.score,  # weighted score
                match  # store the match object
            )
        
        # Process vector results and combine with keyword results
        for match in vector_result.matches:
            if match.document_id in combined_scores:
                # Update existing entry
                keyword_score, _, _, keyword_match = combined_scores[match.document_id]
                combined_score = (1 - self.hybrid_alpha) * keyword_score + self.hybrid_alpha * match.score
                
                # Keep the match with the higher score or the vector match if chunk_id matches
                if keyword_match.chunk_id == match.chunk_id or match.score > keyword_score:
                    final_match = match
                else:
                    final_match = keyword_match
                
                combined_scores[match.document_id] = (
                    keyword_score,
                    match.score,
                    combined_score,
                    final_match
                )
            else:
                # New entry (no keyword match)
                combined_scores[match.document_id] = (
                    0.0,
                    match.score,
                    self.hybrid_alpha * match.score,
                    match
                )
        
        # Sort by combined score and limit to top_k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1][2],  # sort by combined score
            reverse=True
        )[:query.top_k]
        
        # Create the final matches list
        matches = []
        for doc_id, (keyword_score, vector_score, combined_score, match) in sorted_results:
            # Create a new match with the combined score
            updated_match = DocumentMatch(
                document_id=match.document_id,
                chunk_id=match.chunk_id,
                score=combined_score,
                text=match.text,
                document=match.document,
                chunk=match.chunk,
                metadata={
                    **match.metadata,
                    "keyword_score": keyword_score,
                    "vector_score": vector_score,
                }
            )
            matches.append(updated_match)
        
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
            total_found=len(combined_scores),
            execution_time_ms=execution_time_ms,
            metadata={
                "index_type": "HybridIndex",
                "hybrid_alpha": self.hybrid_alpha,
                "keyword_time_ms": keyword_result.execution_time_ms,
                "vector_time_ms": vector_result.execution_time_ms,
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
        
        # Create subdirectories for the component indexes
        keyword_dir = os.path.join(directory, "keyword_index")
        vector_dir = os.path.join(directory, "vector_index")
        os.makedirs(keyword_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)
        
        # Persist the component indexes
        self.keyword_index.persist(keyword_dir)
        self.vector_index.persist(vector_dir)
        
        # Save documents
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
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
            "hybrid_alpha": self.hybrid_alpha,
        }
        
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, directory: str, embedding_function: Optional[Callable[[str], List[float]]] = None) -> "HybridIndex":
        """
        Load an index from disk.
        
        Args:
            directory: The directory to load the index from
            embedding_function: Optional embedding function to use for the vector index
            
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
            hybrid_alpha=metadata["hybrid_alpha"],
        )
        
        # Set timestamps
        index.created_at = datetime.fromisoformat(metadata["created_at"])
        index.updated_at = datetime.fromisoformat(metadata["updated_at"])
        
        # Set stats
        index._stats = metadata["stats"]
        
        # Load component indexes
        keyword_dir = os.path.join(directory, "keyword_index")
        vector_dir = os.path.join(directory, "vector_index")
        
        index.keyword_index = InvertedIndex.load(keyword_dir)
        index.vector_index = VectorIndex.load(vector_dir, embedding_function)
        
        # Load documents
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            index.documents = pickle.load(f)
        
        # Set document IDs
        index.document_ids = set(metadata["document_ids"])
        
        return index
    
    def process_embedding_queue(self) -> int:
        """
        Process any documents or chunks waiting for embeddings.
        
        This requires an embedding function to be set.
        
        Returns:
            Number of items processed
        """
        return self.vector_index.process_embedding_queue()
    
    def set_embedding_function(self, embedding_function: Callable[[str], List[float]]) -> None:
        """
        Set the embedding function for the vector index.
        
        Args:
            embedding_function: Function that takes text and returns embedding vector
        """
        self.vector_index.set_embedding_function(embedding_function)
    
    def set_hybrid_alpha(self, alpha: float) -> None:
        """
        Set the hybrid weighting parameter.
        
        Args:
            alpha: Weight for combining keyword and semantic scores (0.0-1.0)
                0.0 = pure keyword search, 1.0 = pure semantic search
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("hybrid_alpha must be between 0.0 and 1.0")
        self.hybrid_alpha = alpha 