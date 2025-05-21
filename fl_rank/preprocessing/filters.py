# fl_rank/preprocessing/filters.py
"""
Filter implementations for tag preprocessing.
"""

from typing import List, Optional, Set

from fl_rank.preprocessing.base import Filter


class StopwordFilter(Filter):
    """
    Filter that removes stopwords from tokens.
    """
    
    def __init__(self, stopwords: Optional[Set[str]] = None):
        """
        Initialize the filter.
        
        Args:
            stopwords: Set of stopwords to filter out
        """
        self.stopwords = stopwords or set()
    
    def filter(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by removing stopwords.
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            List[str]: Filtered tokens
        """
        return [token for token in tokens if token.lower() not in self.stopwords]


class MinLengthFilter(Filter):
    """
    Filter that removes tokens shorter than a minimum length.
    """
    
    def __init__(self, min_length: int = 2):
        """
        Initialize the filter.
        
        Args:
            min_length: Minimum token length to keep
        """
        self.min_length = min_length
    
    def filter(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            List[str]: Filtered tokens
        """
        return [token for token in tokens if len(token) >= self.min_length]


class DefaultFilter(Filter):
    """
    Default filter that combines multiple filtering strategies.
    """
    
    def __init__(self):
        """
        Initialize the default filter.
        """
        self.min_length = MinLengthFilter(min_length=1)  # Default: keep tokens of any length
        
        # Common technical stopwords that don't add semantic value
        common_stopwords = {"the", "and", "or", "in", "of", "a", "to", "for"}
        self.stopwords = StopwordFilter(stopwords=common_stopwords)
    
    def filter(self, tokens: List[str]) -> List[str]:
        """
        Apply all filtering strategies in sequence.
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            List[str]: Filtered tokens
        """
        # Apply stopword filter first
        filtered = self.stopwords.filter(tokens)
        
        # Then apply minimum length filter
        filtered = self.min_length.filter(filtered)
        
        return filtered
