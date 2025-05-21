# fl_rank/preprocessing/normalizers.py
"""
Normalizer implementations for tag preprocessing.
"""

import re
from typing import List

from fl_rank.preprocessing.base import Normalizer


class LowercaseNormalizer(Normalizer):
    """
    Normalizer that converts tokens to lowercase.
    """
    
    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens by converting to lowercase.
        
        Args:
            tokens: List of tokens to normalize
            
        Returns:
            List[str]: Lowercase tokens
        """
        return [token.lower() for token in tokens]


class VersionStripNormalizer(Normalizer):
    """
    Normalizer that strips version numbers from tokens.
    
    For example, "Python3.7" -> "python".
    """
    
    def __init__(self, add_base_form: bool = True):
        """
        Initialize the normalizer.
        
        Args:
            add_base_form: Whether to add the base form alongside the original
        """
        self.add_base_form = add_base_form
        self.version_pattern = re.compile(r'([a-zA-Z]+)[\d\.]+')
    
    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Normalize tokens by stripping version numbers.
        
        Args:
            tokens: List of tokens to normalize
            
        Returns:
            List[str]: Normalized tokens
        """
        result = []
        for token in tokens:
            result.append(token)
            
            if self.add_base_form:
                match = self.version_pattern.match(token)
                if match:
                    base_name = match.group(1).lower()
                    if base_name not in result:
                        result.append(base_name)
        
        return result


class DefaultNormalizer(Normalizer):
    """
    Default normalizer that combines multiple normalization strategies.
    """
    
    def __init__(self):
        """
        Initialize the default normalizer.
        """
        self.lowercase = LowercaseNormalizer()
        self.version_strip = VersionStripNormalizer()
    
    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Apply all normalization strategies in sequence.
        
        Args:
            tokens: List of tokens to normalize
            
        Returns:
            List[str]: Normalized tokens
        """
        # Apply lowercase first
        lowercased = self.lowercase.normalize(tokens)
        
        # Then strip versions
        normalized = self.version_strip.normalize(lowercased)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for token in normalized:
            if token not in seen:
                seen.add(token)
                result.append(token)
        
        return result
