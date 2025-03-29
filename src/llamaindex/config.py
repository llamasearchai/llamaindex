"""
Configuration module for LlamaIndex
"""
import os
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from dataclasses import dataclass, field

class VectorStoreType(str, Enum):
    """Supported vector store types."""
    FAISS = "faiss"
    HNSW = "hnsw"
    ANNOY = "annoy"
    FLAT = "flat"

class DistanceMetric(str, Enum):
    """Supported distance metrics for vector search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"

@dataclass
class IndexConfig:
    """
    Configuration for LlamaIndex.
    
    This class contains all the configuration options for LlamaIndex.
    """
    
    # Vector store configuration
    vector_store_type: VectorStoreType = VectorStoreType.HNSW
    vector_dimensions: int = 768
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # HNSW specific parameters
    hnsw_m: int = 64  # Number of bidirectional links
    hnsw_ef_construction: int = 200  # Size of dynamic candidate list during index building
    hnsw_ef_search: int = 128  # Size of dynamic candidate list during search
    
    # FAISS specific parameters
    faiss_nlist: int = 100  # Number of cells for FAISS IVF
    faiss_nprobe: int = 10  # Number of cells to visit during search
    
    # Annoy specific parameters
    annoy_n_trees: int = 100  # Number of trees for Annoy index
    
    # Chunking configuration
    chunk_size: int = 512
    chunk_overlap: int = 128
    
    # Search configuration
    top_k: int = 10
    min_score: float = 0.0
    
    # Hybrid search configuration
    hybrid_search: bool = False
    hybrid_alpha: float = 0.5  # Weight between vector and keyword search
    
    # Embedding configuration
    embedding_model: str = "mlx"  # Options: "mlx", "sentence-transformers", "custom"
    embedding_batch_size: int = 32
    
    # Storage configuration
    storage_dir: str = os.path.expanduser("~/.llamaindex")
    persist_directory: str = field(init=False)
    
    # Multithreading configuration
    use_threads: bool = True
    max_threads: int = 4
    
    def __post_init__(self):
        """Initialize derived fields."""
        self.persist_directory = os.path.join(self.storage_dir, "indices")
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vector_store_type": self.vector_store_type.value,
            "vector_dimensions": self.vector_dimensions,
            "distance_metric": self.distance_metric.value,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "faiss_nlist": self.faiss_nlist,
            "faiss_nprobe": self.faiss_nprobe,
            "annoy_n_trees": self.annoy_n_trees,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "hybrid_search": self.hybrid_search,
            "hybrid_alpha": self.hybrid_alpha,
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "storage_dir": self.storage_dir,
            "persist_directory": self.persist_directory,
            "use_threads": self.use_threads,
            "max_threads": self.max_threads,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexConfig":
        """Create configuration from dictionary."""
        # Convert string enum values to Enum types
        if "vector_store_type" in data:
            data["vector_store_type"] = VectorStoreType(data["vector_store_type"])
        if "distance_metric" in data:
            data["distance_metric"] = DistanceMetric(data["distance_metric"])
        
        # Create instance with only recognized parameters
        recognized_params = {
            k: v for k, v in data.items()
            if k in IndexConfig.__dataclass_fields__
        }
        
        return cls(**recognized_params) 