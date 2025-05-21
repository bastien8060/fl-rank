# fl_rank/preprocessing/base.py
"""
Base interfaces for tag preprocessing pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union

class Tokenizer(ABC):
    """
    Abstract base class for tag tokenizers.
    
    Tokenizers split compound tags into individual tokens.
    """
    
    @abstractmethod
    def tokenize(self, tags: List[str]) -> List[str]:
        """
        Tokenize a list of tags into individual tokens.
        
        Args:
            tags: List of tags to tokenize
            
        Returns:
            List[str]: Tokenized tags
        """
        pass


class Normalizer(ABC):
    """
    Abstract base class for tag normalizers.
    
    Normalizers standardize tag format (e.g., lowercasing, version stripping).
    """
    
    @abstractmethod
    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Normalize a list of tokens.
        
        Args:
            tokens: List of tokens to normalize
            
        Returns:
            List[str]: Normalized tokens
        """
        pass


class Filter(ABC):
    """
    Abstract base class for tag filters.
    
    Filters remove unwanted tokens (e.g., stopwords).
    """
    
    @abstractmethod
    def filter(self, tokens: List[str]) -> List[str]:
        """
        Filter a list of tokens.
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            List[str]: Filtered tokens
        """
        pass


class Weighter(ABC):
    """
    Abstract base class for tag weighters.
    
    Weighters assign weights to tokens.
    """
    
    @abstractmethod
    def weight(self, tokens: List[str]) -> Dict[str, float]:
        """
        Assign weights to tokens.
        
        Args:
            tokens: List of tokens to weight
            
        Returns:
            Dict[str, float]: Token weights mapping
        """
        pass


class TagPreprocessor(ABC):
    """
    Abstract base class for tag preprocessors.
    
    Tag preprocessors handle the full preprocessing pipeline.
    """
    
    @abstractmethod
    def process(self, tags: List[str]) -> List[str]:
        """
        Process a list of tags through the full pipeline.
        
        Args:
            tags: List of tags to process
            
        Returns:
            List[str]: Processed tags
        """
        pass


class PreprocessingPipeline(TagPreprocessor):
    """
    Default implementation of TagPreprocessor using a configurable pipeline.
    """
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        normalizer: Optional[Normalizer] = None,
        filter_: Optional[Filter] = None,
        weighter: Optional[Weighter] = None
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            tokenizer: Tag tokenizer
            normalizer: Token normalizer
            filter_: Token filter
            weighter: Token weighter
        """
        # Import within method to avoid circular imports
        from fl_rank.preprocessing.tokenizers import DefaultTokenizer
        from fl_rank.preprocessing.normalizers import DefaultNormalizer
        from fl_rank.preprocessing.filters import DefaultFilter
        from fl_rank.preprocessing.weighters import DefaultWeighter
        
        self.tokenizer = tokenizer or DefaultTokenizer()
        self.normalizer = normalizer or DefaultNormalizer()
        self.filter = filter_ or DefaultFilter()
        self.weighter = weighter or DefaultWeighter()
    
    def process(self, tags: List[str]) -> List[str]:
        """
        Process a list of tags through the full pipeline.
        
        Args:
            tags: List of tags to process
            
        Returns:
            List[str]: Processed tags
        """
        tokens = self.tokenizer.tokenize(tags)
        normalized = self.normalizer.normalize(tokens)
        filtered = self.filter.filter(normalized)
        
        # If there are no tokens after filtering, return the original list
        if not filtered and tags:
            return tags
        
        # Apply weighting (future: might affect the returned list order)
        weights = self.weighter.weight(filtered)
        
        # For now, just return the filtered list
        # Future: could return weighted tokens, sorted, etc.
        return filtered