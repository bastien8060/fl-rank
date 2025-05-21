#!/usr/bin/env python3
"""
Simple example showing the core functionality of fl-rank.
"""

from fl_rank import RankingService
from fl_rank.embeddings import SentenceTransformerEmbedding

# 1. Create a simple ranking service with default components
ranking_service = RankingService(embedding_model=SentenceTransformerEmbedding())

# 2. Define some job listings
listings = [
    {
        "id": "job1",
        "title": "Software Engineer",
        "skills": ["python", "javascript", "git"]
    },
    {
        "id": "job2", 
        "title": "Data Scientist",
        "skills": ["python", "statistics", "machine learning"]
    },
    {
        "id": "job3",
        "title": "Web Developer",
        "skills": ["html", "css", "javascript"]
    },
    {
        "id": "job4",
        "title": "Nurse Practitioner",
        "skills": ["patient care", "medical diagnosis", "clinical procedures"]
    }, 
]

# 3. Add listings to the service
print("Adding listings...")
ranking_service.add_items(listings, id_field="id", content_field="skills", metadata_fields=["title"])

# 4. Define a user's skills
user_skills = ["python", "machine learning", "software development"]

# 5. Find matches
print(f"\nFinding matches for user with skills: {user_skills}")
matches = ranking_service.find_similar(user_skills)

# 6. Display results
print("\nTop matches:")
for i, match in enumerate(matches):
    print(f"{i+1}. {match['title']} - Score: {match['score']:.4f}")