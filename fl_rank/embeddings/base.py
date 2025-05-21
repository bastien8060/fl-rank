# fl_rank/embeddings/base.py
"""
Base embedding implementations.
"""

import re
from abc import abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np

from fl_rank.core.base import BaseEmbedding


class TextEmbedding(BaseEmbedding):
    """
    Base class for text embedding models.
    """
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length (L2 normalization).
        
        Args:
            vectors: Array of vectors to normalize
            
        Returns:
            np.ndarray: Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, np.finfo(vectors.dtype).eps)
        return vectors / norms
    
    def process_tags(self, tags: List[str]) -> List[str]:
        """
        Process tags to handle compound tags and versions.
        
        Args:
            tags: List of tags to process
            
        Returns:
            List[str]: Processed tags
        """
        processed_tags = []
        for tag in tags:
            # Split compound tags (e.g., javascript/nodejs)
            parts = tag.lower().split('/')
            processed_tags.extend(parts)
            
            # Handle version tags (e.g., Python3.7)
            for part in parts:
                match = re.match(r'([a-zA-Z]+)[\d\.]+', part)
                if match:
                    base_name = match.group(1).lower()
                    if base_name not in processed_tags:
                        processed_tags.append(base_name)
        
        return processed_tags
    
    def embed_tags(self, tag_lists: List[List[str]], process: bool = True) -> np.ndarray:
        """
        Embed lists of tags into vectors.
        
        Args:
            tag_lists: Lists of tags to embed
            process: Whether to process tags before embedding
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        processed_lists = []
        for tags in tag_lists:
            if process:
                processed_lists.append(self.process_tags(tags))
            else:
                processed_lists.append(tags)
        
        # Flatten for embedding
        all_tags = []
        tag_counts = []
        for tags in processed_lists:
            all_tags.extend(tags)
            tag_counts.append(len(tags))
        
        # Embed all tags at once
        all_embeddings = self.embed(all_tags)
        
        # Compute mean embeddings for each list
        result = []
        start_idx = 0
        for count in tag_counts:
            if count > 0:
                embedding = all_embeddings[start_idx:start_idx + count].mean(axis=0)
                result.append(embedding)
            else:
                # Return zero vector for empty lists
                result.append(np.zeros(self.get_dimension()))
            start_idx += count
        
        return np.array(result)


class PreprocessedVectorEmbedding(BaseEmbedding):
    """
    Embedding class for pre-computed vectors.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize with vector dimension.
        
        Args:
            dimension: Dimension of the pre-computed vectors
        """
        self._dimension = dimension
    
    def embed(self, vectors: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Pass through pre-computed vectors.
        
        Args:
            vectors: Pre-computed vectors
            
        Returns:
            np.ndarray: Array of vectors
        """
        if isinstance(vectors, list):
            return np.array(vectors)
        return vectors
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            int: Dimension of the embedding vectors
        """
        return self._dimension
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length (L2 normalization).
        
        Args:
            vectors: Array of vectors to normalize
            
        Returns:
            np.ndarray: Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, np.finfo(vectors.dtype).eps)
        return vectors / norms