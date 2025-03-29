"""
Index module for LlamaIndex
"""
import os
import json
import time
import pickle
import logging
from typing import List, Dict, Any, Union, Optional, Tuple, Type
import numpy as np
import uuid

from llamaindex.document import Document
from llamaindex.filter import Filter
from llamaindex.results import SearchResult, SearchResults
from llamaindex.config import IndexConfig
from llamaindex.embeddings import Embeddings, MLXEmbeddings

logger = logging.getLogger(__name__)

class Index:
    """
    Main index class for LlamaIndex.
    
    The Index is responsible for indexing documents and searching them.
    """
    
    def __init__(
        self,
        config: Optional[IndexConfig] = None,
        embeddings: Optional[Embeddings] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize an Index.
        
        Args:
            config: Optional index configuration
            embeddings: Optional embeddings model
            name: Optional name for the index
        """
        self.config = config or IndexConfig()
        self.embeddings = embeddings or self._create_default_embeddings()
        self.name = name or f"index-{uuid.uuid4()}"
        
        self.documents: Dict[str, Document] = {}
        self.embeddings_map: Dict[str, List[float]] = {}
        self.vector_store = self._create_vector_store()
        
        logger.info(f"Initialized index '{self.name}' with {self.embeddings.__class__.__name__}")
    
    def _create_default_embeddings(self) -> Embeddings:
        """Create default embeddings based on configuration."""
        if self.config.embedding_model == "mlx":
            try:
                return MLXEmbeddings()
            except ImportError:
                logger.warning("MLX not available, falling back to sentence-transformers")
                self.config.embedding_model = "sentence-transformers"
        
        if self.config.embedding_model == "sentence-transformers":
            try:
                from llamaindex.embeddings import SentenceTransformerEmbeddings
                return SentenceTransformerEmbeddings()
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Install with: pip install sentence-transformers"
                )
        
        raise ValueError(f"Unknown embedding model: {self.config.embedding_model}")
    
    def _create_vector_store(self):
        """Create vector store based on configuration."""
        if self.config.vector_store_type.value == "hnsw":
            try:
                import hnswlib
                
                # Create HNSW index
                vector_store = hnswlib.Index(space='cosine', dim=self.embeddings.dimensions)
                vector_store.init_index(
                    max_elements=10000,  # Will be resized dynamically
                    ef_construction=self.config.hnsw_ef_construction,
                    M=self.config.hnsw_m
                )
                vector_store.set_ef(self.config.hnsw_ef_search)
                
                return vector_store
            except ImportError:
                logger.warning("hnswlib not installed, falling back to flat index")
                self.config.vector_store_type = "flat"
        
        if self.config.vector_store_type.value == "faiss":
            try:
                import faiss
                
                if self.config.distance_metric.value == "cosine":
                    vector_store = faiss.IndexFlatIP(self.embeddings.dimensions)
                else:
                    vector_store = faiss.IndexFlatL2(self.embeddings.dimensions)
                
                return vector_store
            except ImportError:
                logger.warning("faiss not installed, falling back to flat index")
                self.config.vector_store_type = "flat"
        
        # Fallback to flat index (numpy)
        logger.info("Using flat vector store")
        return []  # Will store vectors directly in a list
    
    def add_document(self, document: Document) -> str:
        """
        Add a document to the index.
        
        Args:
            document: Document to add
            
        Returns:
            ID of the added document
        """
        # Store the document
        self.documents[document.id] = document
        
        # Create chunks if needed
        if not document.chunks:
            document.chunk_text(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        # Embed the document
        doc_text = document.text
        embedding = self.embeddings.embed_documents([doc_text])[0]
        self.embeddings_map[document.id] = embedding
        
        # Add to vector store
        self._add_to_vector_store(document.id, embedding)
        
        logger.info(f"Added document {document.id} to index")
        return document.id
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add multiple documents to the index.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        doc_ids = []
        
        # Process in batches for embedding
        for i in range(0, len(documents), self.config.embedding_batch_size):
            batch = documents[i:i + self.config.embedding_batch_size]
            
            # Store documents
            for doc in batch:
                self.documents[doc.id] = doc
                doc_ids.append(doc.id)
                
                # Create chunks if needed
                if not doc.chunks:
                    doc.chunk_text(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap
                    )
            
            # Embed documents
            batch_texts = [doc.text for doc in batch]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            
            # Add to vector store
            for doc, embedding in zip(batch, batch_embeddings):
                self.embeddings_map[doc.id] = embedding
                self._add_to_vector_store(doc.id, embedding)
        
        logger.info(f"Added {len(documents)} documents to index")
        return doc_ids
    
    def _add_to_vector_store(self, doc_id: str, embedding: List[float]):
        """Add an embedding to the vector store."""
        if self.config.vector_store_type.value == "hnsw":
            # Check if we need to resize
            if len(self.documents) > self.vector_store.get_max_elements():
                self.vector_store.resize_index(max(10000, len(self.documents) * 2))
            
            # Add to HNSW index
            self.vector_store.add_items(
                np.array([embedding], dtype=np.float32), 
                np.array([self._get_internal_id(doc_id)])
            )
        
        elif self.config.vector_store_type.value == "faiss":
            # Add to FAISS index
            self.vector_store.add(np.array([embedding], dtype=np.float32))
        
        else:
            # Add to flat index
            self.vector_store.append(embedding)
    
    def _get_internal_id(self, doc_id: str) -> int:
        """Get internal numeric ID for a document ID."""
        # This is a simple implementation - in a real system, you would
        # maintain a proper mapping between string IDs and integer indices
        return list(self.documents.keys()).index(doc_id)
    
    def update_document(self, doc_id: str, document: Document) -> bool:
        """
        Update a document in the index.
        
        Args:
            doc_id: ID of the document to update
            document: Updated document
            
        Returns:
            Whether the update was successful
        """
        if doc_id not in self.documents:
            return False
        
        # The simplest way to update is to remove and re-add
        self.delete_document(doc_id)
        self.add_document(document)
        
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            Whether the deletion was successful
        """
        if doc_id not in self.documents:
            return False
        
        # Remove from documents map
        del self.documents[doc_id]
        
        # Remove from embeddings map
        if doc_id in self.embeddings_map:
            del self.embeddings_map[doc_id]
        
        # For HNSW and FAISS, we can't easily remove items without rebuilding
        # For now, we'll handle this by rebuilding the index if needed
        # In a real implementation, you'd use a more sophisticated approach
        if len(self.documents) > 0 and (
            self.config.vector_store_type.value == "hnsw" or
            self.config.vector_store_type.value == "faiss"
        ):
            self._rebuild_vector_store()
        
        return True
    
    def _rebuild_vector_store(self):
        """Rebuild the vector store from scratch."""
        # Re-initialize vector store
        self.vector_store = self._create_vector_store()
        
        # Re-add all documents
        for doc_id, embedding in self.embeddings_map.items():
            self._add_to_vector_store(doc_id, embedding)
    
    def search(
        self,
        query: str,
        filters: Optional[Filter] = None,
        top_k: Optional[int] = None,
        hybrid_search: Optional[bool] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> SearchResults:
        """
        Search the index.
        
        Args:
            query: Query string
            filters: Optional filters to apply
            top_k: Optional number of results to return (overrides config)
            hybrid_search: Whether to use hybrid search (overrides config)
            hybrid_alpha: Weight between vector and keyword search (overrides config)
            
        Returns:
            Search results
        """
        if not self.documents:
            return SearchResults(results=[], query=query, total=0)
        
        # Use parameters or fall back to config
        top_k = top_k if top_k is not None else self.config.top_k
        use_hybrid = hybrid_search if hybrid_search is not None else self.config.hybrid_search
        alpha = hybrid_alpha if hybrid_alpha is not None else self.config.hybrid_alpha
        
        # Start timing
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search vector store
        doc_ids, vector_scores = self._search_vectors(query_embedding, top_k * 2)
        
        # Hybrid search if requested
        if use_hybrid:
            keyword_doc_ids, keyword_scores = self._keyword_search(query, top_k * 2)
            
            # Combine results with specified alpha
            doc_ids, scores = self._combine_search_results(
                doc_ids, vector_scores,
                keyword_doc_ids, keyword_scores,
                alpha
            )
        else:
            scores = vector_scores
        
        # Apply filters if provided
        if filters:
            doc_ids, scores = self._apply_filters(doc_ids, scores, filters)
        
        # Limit to top_k
        doc_ids = doc_ids[:top_k]
        scores = scores[:top_k]
        
        # Create search results
        results = []
        for doc_id, score in zip(doc_ids, scores):
            document = self.documents[doc_id]
            results.append(SearchResult(document=document, score=score))
        
        # Calculate time taken
        took_ms = (time.time() - start_time) * 1000
        
        return SearchResults(
            results=results,
            query=query,
            total=len(results),
            took_ms=took_ms
        )
    
    def _search_vectors(
        self, query_embedding: List[float], limit: int
    ) -> Tuple[List[str], List[float]]:
        """Search the vector store."""
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        if self.config.vector_store_type.value == "hnsw":
            # Search HNSW index
            internal_ids, distances = self.vector_store.knn_query(
                query_vector.reshape(1, -1), k=min(limit, len(self.documents))
            )
            
            # Convert internal IDs to document IDs
            doc_ids = [list(self.documents.keys())[idx] for idx in internal_ids[0]]
            
            # Convert distances to scores (higher is better)
            scores = [1.0 - dist for dist in distances[0]]
        
        elif self.config.vector_store_type.value == "faiss":
            # Search FAISS index
            distances, internal_ids = self.vector_store.search(
                query_vector.reshape(1, -1), k=min(limit, len(self.documents))
            )
            
            # Convert internal IDs to document IDs
            doc_ids = [list(self.documents.keys())[idx] for idx in internal_ids[0]]
            
            # Convert distances to scores (higher is better)
            if self.config.distance_metric.value == "cosine":
                scores = [float(dist) for dist in distances[0]]
            else:
                # For L2 distance, smaller is better, so invert
                max_dist = max(distances[0]) if len(distances[0]) > 0 else 1.0
                scores = [1.0 - (dist / max_dist) for dist in distances[0]]
        
        else:
            # Search flat index
            doc_ids = []
            scores = []
            
            # Calculate all similarities
            all_ids = list(self.documents.keys())
            all_embeddings = [self.embeddings_map[doc_id] for doc_id in all_ids]
            
            similarities = []
            for embedding in all_embeddings:
                if self.config.distance_metric.value == "cosine":
                    # Cosine similarity
                    similarity = np.dot(query_vector, embedding) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(embedding)
                    )
                else:
                    # Euclidean distance (convert to similarity)
                    distance = np.linalg.norm(query_vector - np.array(embedding))
                    similarity = 1.0 / (1.0 + distance)
                
                similarities.append(similarity)
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1][:limit]
            
            # Get top_k results
            doc_ids = [all_ids[i] for i in sorted_indices]
            scores = [similarities[i] for i in sorted_indices]
        
        return doc_ids, scores
    
    def _keyword_search(
        self, query: str, limit: int
    ) -> Tuple[List[str], List[float]]:
        """Simple keyword search implementation."""
        # This is a very basic keyword search implementation
        # In a real system, you'd use a proper text index
        query_terms = query.lower().split()
        scores = {}
        
        for doc_id, doc in self.documents.items():
            text = doc.text.lower()
            
            # Count term frequencies
            term_count = 0
            for term in query_terms:
                term_count += text.count(term)
            
            # Only include documents that match at least one term
            if term_count > 0:
                # TF-IDF inspired scoring (simplified)
                scores[doc_id] = term_count / len(text.split())
        
        # Sort by score (descending)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Separate IDs and scores
        doc_ids = [item[0] for item in sorted_items]
        score_values = [item[1] for item in sorted_items]
        
        return doc_ids, score_values
    
    def _combine_search_results(
        self,
        vector_doc_ids: List[str],
        vector_scores: List[float],
        keyword_doc_ids: List[str],
        keyword_scores: List[float],
        alpha: float
    ) -> Tuple[List[str], List[float]]:
        """Combine vector and keyword search results."""
        # Combine and normalize scores
        combined_scores = {}
        
        # Normalize vector scores
        if vector_scores:
            max_vector_score = max(vector_scores)
            min_vector_score = min(vector_scores)
            range_vector = max_vector_score - min_vector_score
            
            for doc_id, score in zip(vector_doc_ids, vector_scores):
                if range_vector > 0:
                    normalized_score = (score - min_vector_score) / range_vector
                else:
                    normalized_score = 1.0
                combined_scores[doc_id] = alpha * normalized_score
        
        # Normalize keyword scores
        if keyword_scores:
            max_keyword_score = max(keyword_scores)
            min_keyword_score = min(keyword_scores)
            range_keyword = max_keyword_score - min_keyword_score
            
            for doc_id, score in zip(keyword_doc_ids, keyword_scores):
                if range_keyword > 0:
                    normalized_score = (score - min_keyword_score) / range_keyword
                else:
                    normalized_score = 1.0
                
                if doc_id in combined_scores:
                    combined_scores[doc_id] += (1 - alpha) * normalized_score
                else:
                    combined_scores[doc_id] = (1 - alpha) * normalized_score
        
        # Sort by combined score (descending)
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Separate IDs and scores
        doc_ids = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        return doc_ids, scores
    
    def _apply_filters(
        self, doc_ids: List[str], scores: List[float], filter_: Filter
    ) -> Tuple[List[str], List[float]]:
        """Apply filters to search results."""
        filtered_doc_ids = []
        filtered_scores = []
        
        for doc_id, score in zip(doc_ids, scores):
            if doc_id in self.documents:
                document = self.documents[doc_id]
                if filter_.matches(document.metadata):
                    filtered_doc_ids.append(doc_id)
                    filtered_scores.append(score)
        
        return filtered_doc_ids, filtered_scores
    
    def save(self, directory: Optional[str] = None) -> str:
        """
        Save the index to disk.
        
        Args:
            directory: Optional directory to save to (uses config.persist_directory if not provided)
            
        Returns:
            Path to the saved index
        """
        save_dir = directory or os.path.join(self.config.persist_directory, self.name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save documents
        with open(os.path.join(save_dir, "documents.json"), "w") as f:
            json.dump({
                doc_id: doc.to_dict() for doc_id, doc in self.documents.items()
            }, f)
        
        # Save embeddings
        with open(os.path.join(save_dir, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings_map, f)
        
        # Save vector store (depends on type)
        if self.config.vector_store_type.value == "hnsw":
            self.vector_store.save_index(os.path.join(save_dir, "vector_store.bin"))
        elif self.config.vector_store_type.value == "faiss":
            import faiss
            faiss.write_index(self.vector_store, os.path.join(save_dir, "vector_store.bin"))
        else:
            # Flat index
            with open(os.path.join(save_dir, "vector_store.pkl"), "wb") as f:
                pickle.dump(self.vector_store, f)
        
        # Save config
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)
        
        # Save metadata
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump({
                "name": self.name,
                "embeddings_type": self.embeddings.__class__.__name__,
                "embeddings": self.embeddings.to_dict(),
                "num_documents": len(self.documents),
                "vector_store_type": self.config.vector_store_type.value,
            }, f)
        
        logger.info(f"Saved index to {save_dir}")
        return save_dir
    
    @classmethod
    def load(cls, directory: str) -> "Index":
        """
        Load an index from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded index
        """
        # Load metadata
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Load config
        with open(os.path.join(directory, "config.json"), "r") as f:
            config_dict = json.load(f)
            config = IndexConfig.from_dict(config_dict)
        
        # Create embeddings
        embeddings_type = metadata["embeddings_type"]
        if embeddings_type == "MLXEmbeddings":
            from llamaindex.embeddings import MLXEmbeddings
            embeddings = MLXEmbeddings.from_dict(metadata["embeddings"])
        elif embeddings_type == "SentenceTransformerEmbeddings":
            from llamaindex.embeddings import SentenceTransformerEmbeddings
            embeddings = SentenceTransformerEmbeddings.from_dict(metadata["embeddings"])
        else:
            raise ValueError(f"Unknown embeddings type: {embeddings_type}")
        
        # Create index
        index = cls(config=config, embeddings=embeddings, name=metadata["name"])
        
        # Load documents
        with open(os.path.join(directory, "documents.json"), "r") as f:
            documents_dict = json.load(f)
            index.documents = {
                doc_id: Document.from_dict(doc_dict)
                for doc_id, doc_dict in documents_dict.items()
            }
        
        # Load embeddings
        with open(os.path.join(directory, "embeddings.pkl"), "rb") as f:
            index.embeddings_map = pickle.load(f)
        
        # Load vector store
        if config.vector_store_type.value == "hnsw":
            import hnswlib
            
            # Create HNSW index
            vector_store = hnswlib.Index(space='cosine', dim=embeddings.dimensions)
            vector_store.load_index(
                os.path.join(directory, "vector_store.bin"),
                max_elements=len(index.documents)
            )
            index.vector_store = vector_store
        
        elif config.vector_store_type.value == "faiss":
            import faiss
            
            # Load FAISS index
            index.vector_store = faiss.read_index(os.path.join(directory, "vector_store.bin"))
        
        else:
            # Load flat index
            with open(os.path.join(directory, "vector_store.pkl"), "rb") as f:
                index.vector_store = pickle.load(f)
        
        logger.info(f"Loaded index from {directory} with {len(index.documents)} documents")
        return index

class DistributedIndex:
    """
    Distributed index for LlamaIndex.
    
    This class provides an interface for distributed indexing and search.
    """
    
    def __init__(self, storage_uri: str):
        """
        Initialize a DistributedIndex.
        
        Args:
            storage_uri: URI to the storage backend
        """
        self.storage_uri = storage_uri
        # In a full implementation, this would connect to a distributed storage
        # system like Redis, Elasticsearch, etc.
        raise NotImplementedError(
            "DistributedIndex is not implemented in this version"
        ) 