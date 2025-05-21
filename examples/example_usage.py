# examples/basic_usage.py
"""
Basic usage example for fl-rank.
"""

import time
from pprint import pprint

from fl_rank import RankingService
from fl_rank.embeddings import SentenceTransformerEmbedding
from fl_rank.indexes import FaissIndex
from fl_rank.storage import InMemoryStorage

# Initialize components
embedding = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
index = FaissIndex()
storage = InMemoryStorage()

# Create service
service = RankingService(embedding_model=embedding, index=index, storage=storage)

# Sample data: tech apprenticeships
apprenticeships = [
    {
        "id": "tech1",
        "title": "Software Engineer Apprenticeship",
        "category": "Technology",
        "tags": ["java", "python", "javascript", "testing", "git", "agile", 
                "software development", "algorithms", "data structures"]
    },
    {
        "id": "tech2",
        "title": "Web Developer Apprenticeship",
        "category": "Technology",
        "tags": ["html", "css", "javascript", "react", "nodejs", "frontend", 
                "backend", "responsive design", "UX/UI"]
    },
    {
        "id": "tech3",
        "title": "DevOps Engineer Apprenticeship",
        "category": "Technology",
        "tags": ["AWS", "kubernetes", "docker", "CI/CD", "linux", 
                "infrastructure as code", "automation", "monitoring", "networking"]
    },
    {
        "id": "data1",
        "title": "Data Scientist Apprenticeship",
        "category": "Data",
        "tags": ["python", "R", "statistics", "machine learning", "big data", 
                "data analysis", "SQL", "visualization", "pandas", "pytorch"]
    },
    {
        "id": "data2", 
        "title": "Data Analyst Apprenticeship",
        "category": "Data",
        "tags": ["excel", "SQL", "power BI", "data visualization", "business intelligence", 
                "reporting", "data modeling", "statistics", "tableau"]
    },
    {
        "id": "data3",
        "title": "Machine Learning Engineer Apprenticeship",
        "category": "Data",
        "tags": ["python", "TensorFlow", "deep learning", "NLP", "algorithms", 
                "neural networks", "computer vision", "feature engineering", "model optimization"]
    },
    {
        "id": "med1",
        "title": "Nursing Apprenticeship",
        "category": "Medical",
        "tags": ["patient care", "medical terminology", "vital signs", "pharmacology", 
                "anatomy", "physiology", "clinical skills", "health assessment"]
    },
    {
        "id": "med2",
        "title": "Midwifery Apprenticeship",
        "category": "Medical",
        "tags": ["prenatal care", "childbirth", "postnatal care", "maternal health", 
                "fetal monitoring", "family planning", "women's health", "obstetrics"]
    },
]

# Add apprenticeships to service
print("Adding apprenticeships...")
start_time = time.time()
service.add_items(
    apprenticeships, 
    id_field="id", 
    content_field="tags",
    metadata_fields=["title", "category"]
)
print(f"Added in {time.time() - start_time:.2f} seconds")

# Example user profiles
users = [
    {
        "id": "user1",
        "name": "Alex",
        "tags": ["javascript/nodejs", "Python3.7", "development", "science", 
                "Software Engineer", "data visualisation", "algorithms"]
    },
    {
        "id": "user2",
        "name": "Jamie",
        "tags": ["nursing", "healthcare", "patient care", "medical"]
    }
]

# Find matches for users
print("\nFinding matches for users...")
for user in users:
    print(f"\nUser: {user['name']} ({user['tags']})")
    
    start_time = time.time()
    matches = service.find_similar(user["tags"], k=3)
    query_time = time.time() - start_time
    
    print(f"Top matches (found in {query_time:.4f} seconds):")
    for i, match in enumerate(matches):
        print(f"{i+1}. {match['title']} ({match['category']}) - Score: {match['score']:.4f}")

# Add a new item
print("\nAdding a new apprenticeship...")
new_apprenticeship = {
    "id": "fin1",
    "title": "Financial Analyst Apprenticeship",
    "category": "Finance",
    "tags": ["financial modeling", "excel", "accounting", "forecasting", 
            "budgeting", "data analysis", "valuation"]
}
service.add_items([new_apprenticeship], id_field="id", content_field="tags", metadata_fields=["title", "category"])

# Update an existing item
print("Updating an existing apprenticeship...")
updated_apprenticeship = {
    "id": "tech2",
    "title": "Full-Stack Web Developer Apprenticeship",
    "category": "Technology",
    "tags": ["html", "css", "javascript", "react", "nodejs", "express", "mongodb", 
            "responsive design", "full-stack", "progressive web apps"]
}
service.update_items([updated_apprenticeship], id_field="id", content_field="tags", metadata_fields=["title", "category"])

# Delete an item
print("Deleting an apprenticeship...")
service.delete_items(["med2"])

# Batch querying
print("\nPerforming batch queries...")
start_time = time.time()
batch_results = service.batch_find_similar([user["tags"] for user in users], k=2)
batch_time = time.time() - start_time

print(f"Batch results (found in {batch_time:.4f} seconds):")
for i, (user, results) in enumerate(zip(users, batch_results)):
    print(f"\nUser: {user['name']}")
    for j, match in enumerate(results):
        print(f"{j+1}. {match['title']} ({match['category']}) - Score: {match['score']:.4f}")
