# examples/service_integration.py
"""
Example of how to integrate fl-rank into a service.
"""

import time
import json
from typing import Dict, List, Optional, Any

from fl_rank import RankingService
from fl_rank.embeddings import SentenceTransformerEmbedding, PreprocessedVectorEmbedding
from fl_rank.indexes import FaissIndex, IndexConfig
from fl_rank.storage import InMemoryStorage, StorageConfig

try:
    from fl_rank.storage import PgVectorStorage
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


class RecommendationService:
    """
    Example recommendation service using fl-rank.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize with optional configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.ranking_service = self._initialize_ranking_service()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if config_path is None:
            return {
                "embedding": {
                    "model_name": "all-MiniLM-L6-v2"
                },
                "index": {
                    "type": "flat",
                    "metric": "ip"
                },
                "storage": {
                    "type": "memory"
                }
            }
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _initialize_ranking_service(self) -> RankingService:
        """
        Initialize the ranking service based on configuration.
        
        Returns:
            RankingService: Initialized ranking service
        """
        # Create embedding model
        embedding_config = self.config.get("embedding", {})
        model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
        device = embedding_config.get("device")
        embedding = SentenceTransformerEmbedding(model_name=model_name, device=device)
        
        # Create index
        index_config = self.config.get("index", {})
        index_type = index_config.get("type", "flat")
        metric_type = index_config.get("metric", "ip")
        
        index_conf = IndexConfig(
            index_type=index_type,
            metric_type=metric_type,
            nlist=index_config.get("nlist", 100),
            nprobe=index_config.get("nprobe", 10),
            m=index_config.get("m", 16),
            ef_construction=index_config.get("ef_construction", 100),
            ef_search=index_config.get("ef_search", 50)
        )
        
        index = FaissIndex(config=index_conf)
        
        # Create storage
        storage_config = self.config.get("storage", {})
        storage_type = storage_config.get("type", "memory")
        
        if storage_type == "pgvector" and PGVECTOR_AVAILABLE:
            conn_string = storage_config.get("connection_string")
            table_name = storage_config.get("table_name", "fl_rank_vectors")
            
            storage_conf = StorageConfig(
                connection_string=conn_string,
                table_name=table_name
            )
            
            storage = PgVectorStorage(config=storage_conf)
        else:
            storage = InMemoryStorage()
        
        # Create and initialize ranking service
        service = RankingService(
            embedding_model=embedding,
            index=index,
            storage=storage
        )
        
        service.initialize()
        return service
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the recommendation system.
        
        Args:
            documents: List of documents to add
        """
        self.ranking_service.add_items(
            documents,
            id_field="id",
            content_field="content",
            metadata_fields=["title", "category", "url", "timestamp"]
        )
    
    def get_recommendations(self, user_profile: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        """
        Get recommendations for a user.
        
        Args:
            user_profile: User profile with interests
            k: Number of recommendations to return
            
        Returns:
            List[Dict[str, Any]]: Recommendations
        """
        interests = user_profile.get("interests", [])
        return self.ranking_service.find_similar(interests, k=k)
    
    def get_batch_recommendations(self, user_profiles: List[Dict[str, Any]], k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Get recommendations for multiple users.
        
        Args:
            user_profiles: List of user profiles
            k: Number of recommendations per user
            
        Returns:
            List[List[Dict[str, Any]]]: Recommendations for each user
        """
        interests_list = [profile.get("interests", []) for profile in user_profiles]
        return self.ranking_service.batch_find_similar(interests_list, k=k)
    
    def update_document(self, document: Dict[str, Any]) -> None:
        """
        Update an existing document.
        
        Args:
            document: Document to update
        """
        self.ranking_service.update_items(
            [document],
            id_field="id",
            content_field="content",
            metadata_fields=["title", "category", "url", "timestamp"]
        )
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document.
        
        Args:
            document_id: Document ID to delete
        """
        self.ranking_service.delete_items([document_id])


def demo():
    """
    Run a demo of the recommendation service.
    """
    # Create service
    service = RecommendationService()
    
    # Add some documents
    documents = [
        {
            "id": "doc1",
            "title": "Introduction to Python",
            "content": ["python", "programming", "beginner", "tutorial"],
            "category": "Programming",
            "url": "https://example.com/python-intro",
            "timestamp": "2023-01-15T12:00:00Z"
        },
        {
            "id": "doc2",
            "title": "Advanced Machine Learning",
            "content": ["machine learning", "neural networks", "deep learning", "python", "tensorflow"],
            "category": "Data Science",
            "url": "https://example.com/advanced-ml",
            "timestamp": "2023-02-20T14:30:00Z"
        },
        {
            "id": "doc3",
            "title": "Web Development with React",
            "content": ["web development", "javascript", "react", "frontend", "UI"],
            "category": "Web Development",
            "url": "https://example.com/react-dev",
            "timestamp": "2023-03-10T09:15:00Z"
        },
        {
            "id": "doc4",
            "title": "Medical Advances in Cardiology",
            "content": ["cardiology", "medicine", "heart disease", "research", "treatment"],
            "category": "Medical",
            "url": "https://example.com/cardiology-advances",
            "timestamp": "2023-04-05T16:45:00Z"
        }
    ]
    
    print("Adding documents...")
    service.add_documents(documents)
    
    # Get recommendations for users
    users = [
        {
            "id": "user1",
            "name": "Alice",
            "interests": ["python", "machine learning", "data science"]
        },
        {
            "id": "user2",
            "name": "Bob",
            "interests": ["web development", "javascript", "UI design"]
        },
        {
            "id": "user3",
            "name": "Charlie",
            "interests": ["medicine", "health", "research"]
        }
    ]
    
    print("\nGetting individual recommendations...")
    for user in users:
        print(f"\nRecommendations for {user['name']}:")
        start_time = time.time()
        recs = service.get_recommendations(user, k=2)
        query_time = time.time() - start_time
        
        for i, rec in enumerate(recs):
            print(f"{i+1}. {rec['title']} ({rec['category']}) - Score: {rec['score']:.4f}")
        print(f"Found in {query_time:.4f} seconds")
    
    print("\nGetting batch recommendations...")
    start_time = time.time()
    batch_recs = service.get_batch_recommendations(users, k=2)
    batch_time = time.time() - start_time
    
    for i, (user, recs) in enumerate(zip(users, batch_recs)):
        print(f"\nRecommendations for {user['name']}:")
        for j, rec in enumerate(recs):
            print(f"{j+1}. {rec['title']} ({rec['category']}) - Score: {rec['score']:.4f}")
    print(f"Batch found in {batch_time:.4f} seconds")
    
    # Update a document
    print("\nUpdating a document...")
    updated_doc = {
        "id": "doc3",
        "title": "Modern Web Development with React and Next.js",
        "content": ["web development", "javascript", "react", "next.js", "frontend", "UI", "SSR"],
        "category": "Web Development",
        "url": "https://example.com/react-nextjs-dev",
        "timestamp": "2023-05-12T11:20:00Z"
    }
    service.update_document(updated_doc)
    
    # Delete a document
    print("Deleting a document...")
    service.delete_document("doc4")
    
    # Get recommendations again
    print("\nGetting recommendations after updates:")
    for user in users:
        print(f"\nRecommendations for {user['name']}:")
        recs = service.get_recommendations(user, k=2)
        
        for i, rec in enumerate(recs):
            print(f"{i+1}. {rec['title']} ({rec['category']}) - Score: {rec['score']:.4f}")


if __name__ == "__main__":
    demo()