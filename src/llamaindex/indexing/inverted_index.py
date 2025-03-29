"""
Inverted Index for LlamaIndex.

This module provides an implementation of an inverted index for efficient
keyword-based search.
"""

import os
import json
import pickle
import math
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime

from llamaindex.core.index import Index
from llamaindex.core.document import Document, DocumentChunk
from llamaindex.core.query import Query, QueryResult, QueryMatch
from llamaindex.utils.tokenizer import simple_tokenize, get_default_stopwords


class InvertedIndex(Index):
    """
    An inverted index for efficient keyword-based search.
    
    This index maps terms to the documents and positions where they occur,
    enabling efficient keyword search and relevance ranking using BM25.
    """
    
    def __init__(
        self,
        index_id: Optional[str] = None,
        index_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_idf: bool = True,
        minimum_should_match: float = 0.0,
        case_sensitive: bool = False,
        stopwords: Optional[Set[str]] = None,
    ):
        """
        Initialize an InvertedIndex.
        
        Args:
            index_id: Unique identifier for the index
            index_name: Human-readable name for the index
            metadata: Optional metadata for the index
            use_idf: Whether to use inverse document frequency in scoring
            minimum_should_match: Minimum fraction of query terms that should match
            case_sensitive: Whether to treat terms as case-sensitive
            stopwords: Set of stopwords to ignore during indexing and querying
        """
        super().__init__(index_id, index_name, metadata)
        
        # Index parameters
        self.use_idf = use_idf
        self.minimum_should_match = minimum_should_match
        self.case_sensitive = case_sensitive
        self.stopwords = stopwords or get_default_stopwords()
        
        # Inverted index data structures
        self.doc_lengths: Dict[str, int] = {}  # Document ID -> number of terms
        self.term_docs: Dict[str, Dict[str, List[int]]] = defaultdict(dict)  # Term -> {Doc ID -> [positions]}
        self.term_df: Dict[str, int] = defaultdict(int)  # Term -> document frequency
        
        # Document storage
        self.documents: Dict[str, Document] = {}  # Document ID -> Document
        self.chunks: Dict[str, DocumentChunk] = {}  # Chunk ID -> DocumentChunk
        
        # Statistics
        self.total_terms = 0
        self.avg_doc_length = 0.0
    
    def add_document(self, document: Document) -> None:
        """
        Add a document to the index.
        
        Args:
            document: Document to add to the index
        """
        # Store the document and update doc IDs
        self.documents[document.doc_id] = document
        self.doc_ids.add(document.doc_id)
        
        # Process document chunks, or create a single chunk if none exist
        if not document.chunks:
            # Create a single chunk for the whole document
            chunk = DocumentChunk(
                content=document.content,
                doc_id=document.doc_id,
                metadata=document.metadata.copy()
            )
            document.add_chunk(chunk)
        
        # Index each chunk
        for chunk in document.chunks:
            self._index_chunk(chunk)
        
        # Update statistics
        self._update_statistics()
        self.last_updated = datetime.utcnow()
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            True if the document was deleted, False if it wasn't found
        """
        if doc_id not in self.doc_ids:
            return False
        
        # Get the document
        document = self.documents.get(doc_id)
        if not document:
            return False
        
        # First, remove chunks from the index
        for chunk in document.chunks:
            self._remove_chunk(chunk)
        
        # Remove from storage
        self.doc_ids.remove(doc_id)
        del self.documents[doc_id]
        
        # Update statistics
        self._update_statistics()
        self.last_updated = datetime.utcnow()
        
        return True
    
    def query(self, query: Query) -> QueryResult:
        """
        Query the index.
        
        Args:
            query: Query to execute
            
        Returns:
            QueryResult containing matches and metadata
        """
        start_time = datetime.utcnow()
        
        # Handle empty index case
        if not self.doc_ids:
            return QueryResult(query=query, execution_time_ms=0)
        
        # Tokenize query
        query_terms = self._tokenize(query.query_text)
        
        # Calculate minimum terms to match
        min_terms_to_match = math.ceil(len(query_terms) * self.minimum_should_match)
        
        # Find matching documents and score them
        matches: Dict[str, Tuple[float, Set[str], List[Tuple[str, int, int]]]] = {}  # doc_id -> (score, matched_terms, positions)
        
        # For each term in the query
        for term in query_terms:
            # Skip if term not in index
            if term not in self.term_docs:
                continue
            
            # For each document containing the term
            for doc_id, positions in self.term_docs[term].items():
                # Skip if filters don't match
                if not self._match_filters(doc_id, query.filters):
                    continue
                
                # Get term frequency in document
                tf = len(positions)
                
                # Calculate IDF
                idf = 1.0
                if self.use_idf and self.term_df[term] > 0:
                    idf = math.log(1 + len(self.doc_ids) / self.term_df[term])
                
                # Calculate BM25 score
                k1 = 1.2
                b = 0.75
                dl = self.doc_lengths.get(doc_id, 0)
                avdl = self.avg_doc_length if self.avg_doc_length > 0 else 1.0
                
                tf_part = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl / avdl)) + tf)
                score = idf * tf_part
                
                # Initialize or update document score
                if doc_id not in matches:
                    matches[doc_id] = (0.0, set(), [])
                
                current_score, matched_terms, term_positions = matches[doc_id]
                matches[doc_id] = (
                    current_score + score,
                    matched_terms | {term},
                    term_positions + [(term, pos, pos + len(term)) for pos in positions]
                )
        
        # Filter documents by minimum should match
        qualified_matches = {
            doc_id: (score, matched_terms, positions)
            for doc_id, (score, matched_terms, positions) in matches.items()
            if len(matched_terms) >= min_terms_to_match
        }
        
        # Sort by score descending and take top_k
        sorted_doc_ids = sorted(
            qualified_matches.keys(),
            key=lambda doc_id: qualified_matches[doc_id][0],
            reverse=True
        )[:query.top_k]
        
        # Create QueryMatch objects
        query_matches = []
        for doc_id in sorted_doc_ids:
            score, matched_terms, positions = qualified_matches[doc_id]
            
            # Get the document or chunk content
            if doc_id in self.documents:
                document = self.documents[doc_id]
                content = document.content
                metadata = document.metadata
                chunk_id = None
            else:
                chunk = self.chunks.get(doc_id)
                if not chunk:
                    continue
                content = chunk.content
                metadata = chunk.metadata
                chunk_id = chunk.chunk_id
                doc_id = chunk.doc_id
            
            # Create QueryMatch
            match = QueryMatch(
                doc_id=doc_id,
                content=content,
                metadata=metadata,
                score=score,
                chunk_id=chunk_id,
            )
            query_matches.append(match)
        
        # Calculate execution time
        end_time = datetime.utcnow()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create and return QueryResult
        return QueryResult(
            query=query,
            matches=query_matches,
            total_matches=len(qualified_matches),
            execution_time_ms=execution_time_ms,
        )
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self.doc_lengths.clear()
        self.term_docs.clear()
        self.term_df.clear()
        self.documents.clear()
        self.chunks.clear()
        self.doc_ids.clear()
        self.total_terms = 0
        self.avg_doc_length = 0.0
        self.last_updated = datetime.utcnow()
    
    def persist(self, directory: str) -> str:
        """
        Save the index to disk.
        
        Args:
            directory: Directory to save the index in
            
        Returns:
            Path to the saved index
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save metadata
        self._save_metadata(directory)
        
        # Save index data
        with open(os.path.join(directory, "doc_lengths.pickle"), "wb") as f:
            pickle.dump(self.doc_lengths, f)
        
        with open(os.path.join(directory, "term_docs.pickle"), "wb") as f:
            pickle.dump(dict(self.term_docs), f)
        
        with open(os.path.join(directory, "term_df.pickle"), "wb") as f:
            pickle.dump(dict(self.term_df), f)
        
        with open(os.path.join(directory, "documents.pickle"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(directory, "chunks.pickle"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(os.path.join(directory, "stats.json"), "w") as f:
            json.dump({
                "total_terms": self.total_terms,
                "avg_doc_length": self.avg_doc_length,
            }, f)
        
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({
                "use_idf": self.use_idf,
                "minimum_should_match": self.minimum_should_match,
                "case_sensitive": self.case_sensitive,
                "stopwords": list(self.stopwords),
            }, f)
        
        return directory
    
    @classmethod
    def load(cls, directory: str) -> 'InvertedIndex':
        """
        Load an index from disk.
        
        Args:
            directory: Directory containing the saved index
            
        Returns:
            Loaded index
        """
        # Load configuration
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create index with loaded configuration
        index = cls(
            use_idf=config["use_idf"],
            minimum_should_match=config["minimum_should_match"],
            case_sensitive=config["case_sensitive"],
            stopwords=set(config["stopwords"]),
        )
        
        # Load metadata
        index._load_metadata(directory)
        
        # Load index data
        with open(os.path.join(directory, "doc_lengths.pickle"), "rb") as f:
            index.doc_lengths = pickle.load(f)
        
        with open(os.path.join(directory, "term_docs.pickle"), "rb") as f:
            index.term_docs = defaultdict(dict, pickle.load(f))
        
        with open(os.path.join(directory, "term_df.pickle"), "rb") as f:
            index.term_df = defaultdict(int, pickle.load(f))
        
        with open(os.path.join(directory, "documents.pickle"), "rb") as f:
            index.documents = pickle.load(f)
        
        with open(os.path.join(directory, "chunks.pickle"), "rb") as f:
            index.chunks = pickle.load(f)
        
        # Load statistics
        with open(os.path.join(directory, "stats.json"), "r") as f:
            stats = json.load(f)
            index.total_terms = stats["total_terms"]
            index.avg_doc_length = stats["avg_doc_length"]
        
        return index
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary of index statistics
        """
        stats = super().get_stats()
        stats.update({
            "total_terms": self.total_terms,
            "unique_terms": len(self.term_df),
            "avg_doc_length": self.avg_doc_length,
            "doc_count": len(self.doc_ids),
            "chunk_count": len(self.chunks),
        })
        return stats
    
    def _index_chunk(self, chunk: DocumentChunk) -> None:
        """
        Index a document chunk.
        
        Args:
            chunk: Document chunk to index
        """
        # Store the chunk
        self.chunks[chunk.chunk_id] = chunk
        
        # Tokenize the content
        tokens = self._tokenize(chunk.content)
        
        # Count occurrence positions for each token
        token_positions = defaultdict(list)
        for i, token in enumerate(tokens):
            token_positions[token].append(i)
        
        # Update document length
        self.doc_lengths[chunk.chunk_id] = len(tokens)
        
        # Update term-document index
        for token, positions in token_positions.items():
            # First occurrence of this term in this document
            if chunk.chunk_id not in self.term_docs[token]:
                self.term_df[token] += 1
            
            # Update positions
            self.term_docs[token][chunk.chunk_id] = positions
    
    def _remove_chunk(self, chunk: DocumentChunk) -> None:
        """
        Remove a document chunk from the index.
        
        Args:
            chunk: Document chunk to remove
        """
        # Skip if chunk not in index
        if chunk.chunk_id not in self.chunks:
            return
        
        # Remove from document length index
        if chunk.chunk_id in self.doc_lengths:
            del self.doc_lengths[chunk.chunk_id]
        
        # Remove from term-document index
        for term, docs in list(self.term_docs.items()):
            if chunk.chunk_id in docs:
                del docs[chunk.chunk_id]
                self.term_df[term] -= 1
                
                # Remove term if no more documents
                if self.term_df[term] <= 0:
                    del self.term_df[term]
                    del self.term_docs[term]
        
        # Remove from chunk storage
        del self.chunks[chunk.chunk_id]
    
    def _update_statistics(self) -> None:
        """Update index statistics."""
        # Calculate total terms
        self.total_terms = sum(self.doc_lengths.values())
        
        # Calculate average document length
        num_docs = len(self.doc_lengths)
        self.avg_doc_length = self.total_terms / num_docs if num_docs > 0 else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for indexing or querying.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return simple_tokenize(
            text,
            case_sensitive=self.case_sensitive,
            min_length=1,
            stopwords=self.stopwords,
        )
    
    def _match_filters(self, doc_id: str, filters: Optional[Dict[str, Any]]) -> bool:
        """
        Check if a document matches the query filters.
        
        Args:
            doc_id: Document ID to check
            filters: Filters to apply
            
        Returns:
            True if the document matches the filters, False otherwise
        """
        if not filters:
            return True
        
        # Get document or chunk
        metadata = None
        if doc_id in self.documents:
            metadata = self.documents[doc_id].metadata
        elif doc_id in self.chunks:
            metadata = self.chunks[doc_id].metadata
        
        if not metadata:
            return False
        
        # Check each filter
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                # Check if any value matches
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True 