# fl_rank/service/ranking_service.py
"""
Main service for vector ranking.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fl_rank.core.base import BaseEmbedding, BaseIndex, BaseStorage
from fl_rank.embeddings import SentenceTransformerEmbedding
from fl_rank.indexes import FaissIndex
from fl_rank.storage import InMemoryStorage
from fl_rank.utils.vector import normalize_vectors
from fl_rank.preprocessing import PreprocessingPipeline, TagPreprocessor
from fl_rank.ranking import Ranker, DefaultRankingStrategy, SimilarityMetric, CosineSimilarity


class DefaultRanker(Ranker):
    """
    Default implementation of Ranker.
    """
    
    def __init__(self, metric: Optional[SimilarityMetric] = None):
        """
        Initialize the ranker.
        
        Args:
            metric: Similarity metric to use
        """
        self.strategy = DefaultRankingStrategy()
    
    def find_similar(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        candidate_ids: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find items similar to query.
        
        Args:
            query_vector: Query vector
            candidate_vectors: Candidate vectors
            candidate_ids: Candidate IDs
            metadata: Optional metadata for candidates
            k: Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Similar items with scores
        """
        return self.strategy.rank(
            query_vector=query_vector,
            candidate_vectors=candidate_vectors,
            candidate_ids=candidate_ids,
            metadata=metadata,
            k=k
        )


class RankingService:
    """
    Service for managing and querying vector rankings.
    """
    
    def __init__(
        self,
        embedding_model: Optional[BaseEmbedding] = None,
        index: Optional[BaseIndex] = None,
        storage: Optional[BaseStorage] = None,
        preprocessor: Optional[TagPreprocessor] = None,
        ranker: Optional[Ranker] = None,
        normalize_vectors: bool = True
    ):
        """
        Initialize the ranking service.
        
        Args:
            embedding_model: Embedding model to use
            index: Vector index to use
            storage: Storage backend to use
            preprocessor: Tag preprocessor to use
            ranker: Ranker to use
            normalize_vectors: Whether to normalize vectors
        """
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()
        self.index = index or FaissIndex()
        self.storage = storage or InMemoryStorage()
        self.preprocessor = preprocessor or PreprocessingPipeline()
        self.ranker = ranker or DefaultRanker()
        self.normalize_vectors = normalize_vectors
        self.is_initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the service with existing data from storage.
        """
        # Try to load index from storage
        index = self.storage.get_index()
        if index is not None:
            self.index = index
            self.is_initialized = True
            return
        
        # Otherwise, build index from stored vectors
        ids, vectors, _ = self.storage.get_vectors()
        if len(ids) > 0:
            self.index.build(vectors, ids)
            self.is_initialized = True
    
    def add_items(
        self, 
        items: List[Dict[str, Any]], 
        id_field: str = "id", 
        content_field: str = "content",
        metadata_fields: Optional[List[str]] = None
    ) -> None:
        """
        Add items to the ranking system.
        
        Args:
            items: List of items to add
            id_field: Field name for item IDs
            content_field: Field name for content to embed
            metadata_fields: Optional list of fields to store as metadata
        """
        if not items:
            return
        
        # Extract content and IDs
        contents = []
        ids = []
        metadata_list = []
        
        for item in items:
            # Get or generate ID
            id_ = item.get(id_field, str(uuid.uuid4()))
            ids.append(id_)
            
            # Get content to embed
            content = item.get(content_field)
            if isinstance(content, list):
                # Already a list of tags/texts
                contents.append(content)
            elif isinstance(content, str):
                # Single text
                contents.append([content])
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
            
            # Extract metadata
            if metadata_fields:
                metadata = {field: item.get(field) for field in metadata_fields if field in item}
            else:
                metadata = {}
            metadata_list.append(metadata)
        
        # Generate embeddings
        start_time = time.time()
        vectors = self._embed_contents(contents)
        print(f"Embedded {len(contents)} items in {time.time() - start_time:.2f} seconds")
        
        # Store vectors and metadata
        self.storage.store_vectors(ids, vectors, metadata_list)
        
        # Update or build index
        if self.is_initialized:
            self.index.add(vectors, ids)
        else:
            self.index.build(vectors, ids)
            self.is_initialized = True
        
        # Store updated index
        self.storage.store_index(self.index)
    
    def update_items(
        self,
        items: List[Dict[str, Any]],
        id_field: str = "id",
        content_field: str = "content",
        metadata_fields: Optional[List[str]] = None
    ) -> None:
        """
        Update existing items.
        
        Args:
            items: List of items to update
            id_field: Field name for item IDs
            content_field: Field name for content to embed
            metadata_fields: Optional list of fields to store as metadata
        """
        # Extract existing IDs
        ids = [item[id_field] for item in items if id_field in item]
        if not ids:
            return
        
        # Delete existing items
        self.delete_items(ids)
        
        # Add as new items
        self.add_items(items, id_field, content_field, metadata_fields)
    
    def delete_items(self, item_ids: List[Any]) -> None:
        """
        Delete items from the ranking system.
        
        Args:
            item_ids: List of item IDs to delete
        """
        if not item_ids:
            return
        
        # Delete from storage
        self.storage.delete_vectors(item_ids)
        
        # Rebuild index from remaining vectors
        ids, vectors, _ = self.storage.get_vectors()
        if len(ids) > 0:
            self.index.build(vectors, ids)
        else:
            # No vectors left
            self.index = FaissIndex()
        
        # Store updated index
        self.storage.store_index(self.index)
    
    def find_similar(
        self, 
        query: Union[List[str], np.ndarray, Dict[str, Any]], 
        k: int = 10,
        content_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Find items similar to a query.
        
        Args:
            query: Query to find similar items for (text list, vector, or dict)
            k: Number of results to return
            content_field: Field name for content if query is a dict
            
        Returns:
            List[Dict[str, Any]]: List of similar items with scores
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self.is_initialized:
            return []
        
        # Convert query to a vector
        query_vector = self._prepare_query_vector(query, content_field)
        
        # Get all vectors from storage
        ids, vectors, metadata_list = self.storage.get_vectors()
        
        # Use ranker to find similar items
        results = self.ranker.find_similar(
            query_vector=query_vector,
            candidate_vectors=vectors,
            candidate_ids=ids,
            metadata=metadata_list,
            k=k
        )
        
        return results
    
    def batch_find_similar(
        self,
        queries: List[Union[List[str], np.ndarray, Dict[str, Any]]],
        k: int = 10,
        content_field: str = "content"
    ) -> List[List[Dict[str, Any]]]:
        """
        Find similar items for multiple queries.
        
        Args:
            queries: List of queries
            k: Number of results to return per query
            content_field: Field name for content if queries are dicts
            
        Returns:
            List[List[Dict[str, Any]]]: List of result lists
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self.is_initialized:
            return [[] for _ in queries]
        
        # Get all vectors from storage
        ids, vectors, metadata_list = self.storage.get_vectors()
        
        results = []
        for query in queries:
            # Convert query to a vector
            query_vector = self._prepare_query_vector(query, content_field)
            
            # Use ranker to find similar items
            query_results = self.ranker.find_similar(
                query_vector=query_vector,
                candidate_vectors=vectors,
                candidate_ids=ids,
                metadata=metadata_list,
                k=k
            )
            
            results.append(query_results)
        
        return results
    
    def get_item(self, item_id: Any) -> Optional[Dict[str, Any]]:
        """
        Get a specific item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            Optional[Dict[str, Any]]: Item if found, None otherwise
        """
        ids, _, metadata_list = self.storage.get_vectors([item_id])
        if not ids:
            return None
        
        result = {"id": ids[0]}
        result.update(metadata_list[0])
        return result
    
    def get_items(self, item_ids: List[Any]) -> List[Dict[str, Any]]:
        """
        Get multiple items by IDs.
        
        Args:
            item_ids: List of item IDs
            
        Returns:
            List[Dict[str, Any]]: List of found items
        """
        ids, _, metadata_list = self.storage.get_vectors(item_ids)
        
        results = []
        for i, id_ in enumerate(ids):
            result = {"id": id_}
            result.update(metadata_list[i])
            results.append(result)
        
        return results
    
    def _embed_contents(self, contents: List[List[str]]) -> np.ndarray:
        """
        Embed multiple content lists.
        
        Args:
            contents: List of content lists to embed
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        # Process tags through the preprocessing pipeline
        processed_contents = []
        for content in contents:
            processed_content = self.preprocessor.process(content)
            processed_contents.append(processed_content)
        
        # Embed the processed content
        all_tags = []
        tag_counts = []
        
        for tags in processed_contents:
            all_tags.extend(tags)
            tag_counts.append(len(tags))
        
        if not all_tags:
            # Handle empty tags case
            dimension = self.embedding_model.get_dimension()
            return np.zeros((len(contents), dimension))
        
        # Embed all tags
        all_embeddings = self.embedding_model.embed(all_tags)
        
        # Compute mean embeddings for each list
        vectors = []
        start_idx = 0
        for count in tag_counts:
            if count > 0:
                embedding = all_embeddings[start_idx:start_idx + count].mean(axis=0)
                vectors.append(embedding)
            else:
                # Return zero vector for empty lists
                vectors.append(np.zeros(self.embedding_model.get_dimension()))
            start_idx += count
        
        vectors = np.array(vectors)
        
        # Normalize if required
        if self.normalize_vectors:
            vectors = normalize_vectors(vectors)
        
        return vectors
    
    def _prepare_query_vector(
        self, 
        query: Union[List[str], np.ndarray, Dict[str, Any]], 
        content_field: str
    ) -> np.ndarray:
        """
        Prepare a query vector from various input types.
        
        Args:
            query: Query input
            content_field: Field name for content if query is a dict
            
        Returns:
            np.ndarray: Query vector
        """
        if isinstance(query, np.ndarray):
            # Already a vector
            query_vector = query
        elif isinstance(query, list):
            # List of strings
            if len(query) == 0:
                raise ValueError("Empty query list")
            if isinstance(query[0], (int, float)):
                # Numeric vector
                query_vector = np.array(query)
            else:
                # List of texts/tags
                processed_query = self.preprocessor.process(query)
                query_vector = self._embed_contents([processed_query])[0]
        elif isinstance(query, dict):
            # Dictionary with content field
            content = query.get(content_field)
            if content is None:
                raise ValueError(f"Query dict missing '{content_field}' field")
            if isinstance(content, list):
                processed_content = self.preprocessor.process(content)
                query_vector = self._embed_contents([processed_content])[0]
            else:
                processed_content = self.preprocessor.process([content])
                query_vector = self._embed_contents([processed_content])[0]
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")
        
        # Ensure vector is properly shaped
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        return query_vector