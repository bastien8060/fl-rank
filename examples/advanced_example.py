# examples/advanced_example.py
"""
Advanced usage example for the refactored fl-rank.
"""

import time
from pprint import pprint
import numpy as np

from fl_rank import RankingService
from fl_rank.embeddings import SentenceTransformerEmbedding
from fl_rank.indexes import FaissIndex
from fl_rank.storage import InMemoryStorage
from fl_rank.preprocessing import (
    PreprocessingPipeline,
    CompoundTagTokenizer,
    LowercaseNormalizer,
    VersionStripNormalizer,
    StopwordFilter,
    UniformWeighter
)
from fl_rank.ranking import (
    DefaultRanker,
    CosineSimilarity,
    EuclideanDistance,
    WeightedRankingStrategy
)

# Define job listings
listings = [
    {
        "id": "job1",
        "title": "Software Engineer",
        "category": "Technology",
        "skills": ["python", "nodejs/javascript", "React.js", "frontend", "github", "algos", "data structures"],
        "experience": "5+ years",
        "importance": 10,
    },
    {
        "id": "job2", 
        "title": "Web Developer",
        "category": "Technology",
        "skills": ["html", "css", "web design", "react", "nodejs"],
        "experience": "2+ years",
        "importance": 6,
    },
    {
        "id": "job3",
        "title": "Data Scientist",
        "category": "Data",
        "skills": ["python3.8", "statistics", "golang", "machine learning", "data analysis", "pandas"],
        "experience": "3+ years",
        "importance": 8,
    },
    {
        "id": "job4",
        "title": "DevOps Engineer",
        "category": "Technology",
        "skills": ["aws", "kubernetes", "docker", "linux", "ci/cd"],
        "experience": "4+ years",
        "importance": 9,
    },
    {
        "id": "job5",
        "title": "Nurse Practitioner",
        "category": "Medical",
        "skills": ["patient care", "medical diagnosis", "pharmacology", "clinical procedures"],
        "experience": "7+ years",
        "importance": 7,
    }
]

# Define user profile
user_skills = ["javascript", "Python3.7", "ReactJS", "data visualization", "algorithms"]

print("===== BASIC RANKING WITH DEFAULT SETTINGS =====")

# Create a basic service with defaults
basic_service = RankingService(
    embedding_model=SentenceTransformerEmbedding(),
    index=FaissIndex()
)

# Add listings
basic_service.add_items(
    items=listings, 
    id_field="id", 
    content_field="skills", 
    metadata_fields=["title", "category", "importance"]
)

# Find matches
basic_matches = basic_service.find_similar(user_skills, k=len(listings))

# Print basic results
print("\nDefault Ranking Results:")
for i, match in enumerate(basic_matches):
    print(f"{i+1}. {match['title']} ({match['category']}) - Score: {match['score']:.4f}")

print("\n===== CUSTOM PREPROCESSING PIPELINE =====")

# Create custom preprocessing pipeline
custom_preprocessor = PreprocessingPipeline(
    tokenizer=CompoundTagTokenizer(separator="/"),
    normalizer=VersionStripNormalizer(add_base_form=True),
    filter_=StopwordFilter(stopwords={"and", "or", "the"}),
    weighter=UniformWeighter()
)

# Create service with custom preprocessing
preprocessor_service = RankingService(
    embedding_model=SentenceTransformerEmbedding(),
    index=FaissIndex(),
    preprocessor=custom_preprocessor
)

# Add listings
preprocessor_service.add_items(
    items=listings, 
    id_field="id", 
    content_field="skills", 
    metadata_fields=["title", "category", "importance"]
)

# Find matches
preprocessor_matches = preprocessor_service.find_similar(user_skills, k=len(listings))

# Print results with custom preprocessing
print("\nResults with Custom Preprocessing:")
for i, match in enumerate(preprocessor_matches):
    print(f"{i+1}. {match['title']} ({match['category']}) - Score: {match['score']:.4f}")

print("\n===== ALTERNATIVE SIMILARITY METRICS =====")

# Create service with Euclidean distance metric
euclidean_metric = EuclideanDistance()
euclidean_ranker = DefaultRanker(metric=euclidean_metric)
euclidean_service = RankingService(
    embedding_model=SentenceTransformerEmbedding(),
    index=FaissIndex(),
    ranker=euclidean_ranker
)

# Add listings
euclidean_service.add_items(
    items=listings, 
    id_field="id", 
    content_field="skills", 
    metadata_fields=["title", "category", "importance"]
)

# Find matches
euclidean_matches = euclidean_service.find_similar(user_skills, k=len(listings))

# Print results with Euclidean distance
print("\nResults with Euclidean Distance:")
for i, match in enumerate(euclidean_matches):
    print(f"{i+1}. {match['title']} ({match['category']}) - Score: {match['score']:.4f}")

print("\n===== WEIGHTED RANKING STRATEGY =====")

# Define a weight function based on job importance
def weight_by_importance(metadata):
    importance = metadata.get("importance", 5)
    return importance / 10.0  # Normalize to 0-1 range

# Create a weighted ranking strategy
weighted_strategy = WeightedRankingStrategy(
    metric=CosineSimilarity(),
    weight_fn=weight_by_importance
)

# Create a ranker with the weighted strategy
weighted_ranker = DefaultRanker()
weighted_ranker.strategy = weighted_strategy

# Create service with weighted ranking
weighted_service = RankingService(
    embedding_model=SentenceTransformerEmbedding(),
    index=FaissIndex(),
    ranker=weighted_ranker
)

# Add listings
weighted_service.add_items(
    items=listings, 
    id_field="id", 
    content_field="skills", 
    metadata_fields=["title", "category", "importance"]
)

# Find matches
weighted_matches = weighted_service.find_similar(user_skills, k=len(listings))

# Print results with weighted ranking
print("\nResults with Weighted Ranking:")
for i, match in enumerate(weighted_matches):
    importance = match.get("importance", 5)
    print(f"{i+1}. {match['title']} ({match['category']}) - " +
          f"Base Score: {match['score']:.4f}, " +
          f"Importance: {importance}, " +
          f"Weighted Score: {match['weighted_score']:.4f}")

print("\n===== COMPARISON OF ALL APPROACHES =====")

# Create a comparison table
print("\nRanking Comparison:")
print("-" * 80)
print(f"{'Job Title':<25} | {'Default':<10} | {'Custom Preproc':<15} | {'Euclidean':<10} | {'Weighted':<10}")
print("-" * 80)

# Map results by job ID
basic_map = {m["id"]: (i+1, m["score"]) for i, m in enumerate(basic_matches)}
preproc_map = {m["id"]: (i+1, m["score"]) for i, m in enumerate(preprocessor_matches)}
euclidean_map = {m["id"]: (i+1, m["score"]) for i, m in enumerate(euclidean_matches)}
weighted_map = {m["id"]: (i+1, m.get("weighted_score", m["score"])) for i, m in enumerate(weighted_matches)}

# Print comparison for each job
for listing in listings:
    job_id = listing["id"]
    title = listing["title"]
    
    basic_rank, basic_score = basic_map.get(job_id, (0, 0.0))
    preproc_rank, preproc_score = preproc_map.get(job_id, (0, 0.0))
    euclidean_rank, euclidean_score = euclidean_map.get(job_id, (0, 0.0))
    weighted_rank, weighted_score = weighted_map.get(job_id, (0, 0.0))
    
    print(f"{title:<25} | #{basic_rank:<2} ({basic_score:.2f}) | " +
          f"#{preproc_rank:<2} ({preproc_score:.2f}) | " +
          f"#{euclidean_rank:<2} ({euclidean_score:.2f}) | " +
          f"#{weighted_rank:<2} ({weighted_score:.2f})")

print("-" * 80)
print("\nConclusions:")
print("1. The custom preprocessing helps with mapping versions and compound tags")
print("2. Different similarity metrics can produce different orderings")
print("3. Weighted ranking can prioritize specific items based on metadata")