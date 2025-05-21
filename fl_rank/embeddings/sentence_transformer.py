# fl_rank/embeddings/sentence_transformer.py
"""
Sentence Transformer embedding model.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from fl_rank.embeddings.base import TextEmbedding


class SentenceTransformerEmbedding(TextEmbedding):
    """
    Embedding model using Sentence Transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize with model name and optional device.
        
        Args:
            model_name: Name of the Sentence Transformer model
            device: Device to use (e.g., 'cuda', 'cpu', None for auto)
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embedding vectors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        return self.model.encode(texts, normalize_embeddings=True)
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            int: Dimension of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()