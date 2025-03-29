"""
SentenceTransformer embeddings for LlamaIndex
"""
from typing import List, Dict, Any, Union, Optional
import logging

from llamaindex.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    """
    SentenceTransformer embeddings for LlamaIndex.
    
    This class uses the sentence-transformers library to create embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize SentenceTransformerEmbeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on (e.g. "cpu", "cuda", "mps")
            batch_size: Batch size for embedding multiple texts
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        
        try:
            self._initialize_model()
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    
    def _initialize_model(self):
        """Initialize the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        logger.info(f"Initializing SentenceTransformer model: {self._model_name}")
        
        self._model = SentenceTransformer(self._model_name, device=self._device)
        
        logger.info(
            f"SentenceTransformer model initialized successfully on "
            f"device: {self._model.device}"
        )
    
    @property
    def dimensions(self) -> int:
        """Return the dimensions of the embeddings."""
        return self._model.get_sentence_embedding_dimension()
    
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
        
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            The query embedding
        """
        return self.embed_documents([text])[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert embeddings to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the embeddings
        """
        return {
            "type": self.__class__.__name__,
            "model_name": self._model_name,
            "device": self._device,
            "batch_size": self._batch_size,
            "dimensions": self.dimensions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformerEmbeddings":
        """
        Create SentenceTransformerEmbeddings from a dictionary.
        
        Args:
            data: Dictionary representation of embeddings
            
        Returns:
            SentenceTransformerEmbeddings instance
        """
        return cls(
            model_name=data.get("model_name", "all-MiniLM-L6-v2"),
            device=data.get("device"),
            batch_size=data.get("batch_size", 32),
        ) 