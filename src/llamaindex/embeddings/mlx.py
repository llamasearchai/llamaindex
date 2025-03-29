"""
MLX embeddings for LlamaIndex, optimized for Apple Silicon
"""
import os
from typing import List, Dict, Any, Union, Optional
import logging

from llamaindex.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class MLXEmbeddings(Embeddings):
    """
    MLX-powered embeddings for Apple Silicon devices.
    
    This class uses MLX to run embeddings efficiently on Apple Silicon.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        dimensions: int = 384,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize MLXEmbeddings.
        
        Args:
            model_name: Name of the embedding model to use
            dimensions: Embedding dimensions
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for embedding multiple texts
        """
        self._model_name = model_name
        self._dimensions = dimensions
        self._normalize = normalize
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None
        
        try:
            self._initialize_model()
        except ImportError:
            logger.warning(
                "MLX not available. Make sure you are running on Apple Silicon "
                "and have installed MLX: pip install mlx"
            )
            raise
    
    def _initialize_model(self):
        """Initialize the MLX model and tokenizer."""
        try:
            import mlx.core as mx
            from transformers import AutoTokenizer
            from mlx.transformers import AutoModel
        except ImportError:
            raise ImportError(
                "MLX or transformers not installed. "
                "Install with: pip install mlx transformers"
            )
        
        logger.info(f"Initializing MLX model: {self._model_name}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        
        # Load model with MLX
        self._model = AutoModel.from_pretrained(self._model_name)
        
        logger.info("MLX model initialized successfully")
    
    @property
    def dimensions(self) -> int:
        """Return the dimensions of the embeddings."""
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
        
        import mlx.core as mx
        import numpy as np
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i:i + self._batch_size]
            
            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="np",
                max_length=512
            )
            
            # Convert to MLX arrays
            input_ids = mx.array(inputs["input_ids"])
            attention_mask = mx.array(inputs["attention_mask"])
            
            # Generate embeddings
            outputs = self._model(input_ids, attention_mask=attention_mask)
            
            # Mean pooling
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
            # Normalize if requested
            if self._normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            # Convert to numpy arrays
            embeddings_np = embeddings.tolist()
            all_embeddings.extend(embeddings_np)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            The query embedding
        """
        return self.embed_documents([text])[0]
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling of token embeddings, weighted by attention mask.
        
        Args:
            token_embeddings: Token embeddings from the model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Pooled embeddings
        """
        import mlx.core as mx
        
        # Convert mask to float and expand dimensions
        input_mask_expanded = attention_mask.reshape(
            attention_mask.shape[0], attention_mask.shape[1], 1
        )
        
        # Sum token embeddings * attention mask
        sum_embeddings = mx.sum(
            token_embeddings * input_mask_expanded, axis=1
        )
        
        # Sum attention mask
        sum_mask = mx.sum(input_mask_expanded, axis=1)
        
        # Average pooling
        return sum_embeddings / sum_mask
    
    def _normalize_embeddings(self, embeddings):
        """
        L2-normalize the embeddings.
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        import mlx.core as mx
        
        # Calculate L2 norm
        norm = mx.sqrt(mx.sum(embeddings * embeddings, axis=1, keepdims=True))
        
        # Normalize
        return embeddings / norm
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert embeddings to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the embeddings
        """
        return {
            "type": self.__class__.__name__,
            "model_name": self._model_name,
            "dimensions": self._dimensions,
            "normalize": self._normalize,
            "batch_size": self._batch_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLXEmbeddings":
        """
        Create MLXEmbeddings from a dictionary.
        
        Args:
            data: Dictionary representation of embeddings
            
        Returns:
            MLXEmbeddings instance
        """
        return cls(
            model_name=data.get("model_name", "intfloat/e5-small-v2"),
            dimensions=data.get("dimensions", 384),
            normalize=data.get("normalize", True),
            batch_size=data.get("batch_size", 32),
        ) 