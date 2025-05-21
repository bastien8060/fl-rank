# simple_job_matcher.py
import json
import time
import numpy as np
from fl_rank import RankingService
from fl_rank.embeddings import SentenceTransformerEmbedding, PreprocessedVectorEmbedding
from fl_rank.indexes import FaissIndex

# 1. Define job listings
listings = [
    {
        "id": "job1",
        "title": "Software Engineer",
        "skills": ["python", "nodejs", "react", "frontend", "github", "algos", "data structures"]
    },
    {
        "id": "job2", 
        "title": "Web Developer",
        "skills": ["html", "css", "web design", "react", "nodejs"]
    },
    {
        "id": "job3",
        "title": "Data Scientist",
        "skills": ["python", "statistics", "golang", "machine learning", "data analysis", "pandas"]
    },
    {
        "id": "job4",
        "title": "DevOps Engineer",
        "skills": ["aws", "kubernetes", "docker", "linux", "ci/cd"]
    },
    {
        "id": "job5",
        "title": "Nurse Practitioner",
        "skills": ["patient care", "medical diagnosis", "pharmacology", "clinical procedures"]
    }
]

# 2. Define user profile
user_skills = ["javascript", "Python3.7", "ReactJS", "data visualization", "algorithms"]

print("=" * 80)
print("PART 1: CREATING AND EXPORTING VECTORS")
print("=" * 80)

# Initialize embedding model
start_time = time.time()
embedding_model = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
model_load_time = time.time() - start_time
print(f"[1.1] Embedding model loaded in {model_load_time:.4f} seconds")

# Create initial service
service = RankingService(
    embedding_model=embedding_model,
    index=FaissIndex()
)

# Add listings to generate embeddings
start_time = time.time()
service.add_items(
    items=listings, 
    id_field="id", 
    content_field="skills", 
    metadata_fields=["title"]
)
embedding_time = time.time() - start_time
print(f"[1.2] Embeddings generated for {len(listings)} listings in {embedding_time:.4f} seconds")

# Extract embeddings and metadata
start_time = time.time()
ids, vectors, metadata_list = service.storage.get_vectors()

# Convert vectors to list format (compatible with Float32Array in EdgeDB/pgvector)
vector_lists = [vector.astype(np.float32).tolist() for vector in vectors]

# Create records for database storage
db_records = []
for i, id_val in enumerate(ids):
    db_records.append({
        "id": id_val,
        "title": metadata_list[i].get("title", ""),
        "vector": vector_lists[i]  # Float32Array compatible
    })

export_time = time.time() - start_time
print(f"[1.3] Vectors prepared for database export in {export_time:.4f} seconds")
print(f"     - Vector dimension: {len(vector_lists[0])}")
print(f"     - Total records: {len(db_records)}")

# Simulate database storage
simulated_db = json.dumps(db_records)
print("[1.4] Vectors exported to database (simulated)")

print("\n\n")
print("=" * 80)
print("PART 2: IMPORTING VECTORS FROM DATABASE AND RANKING")
print("=" * 80)

# In a real scenario, this would be a separate process 
# that loads vectors from the database

# Simulate retrieving from database
print("[2.1] Retrieving vectors from database")
start_time = time.time()
retrieved_records = json.loads(simulated_db)

# Reconstruct vectors and metadata
retrieved_ids = []
retrieved_vectors = []
retrieved_metadata = []

for record in retrieved_records:
    retrieved_ids.append(record["id"])
    # Convert list back to numpy array
    retrieved_vectors.append(np.array(record["vector"], dtype=np.float32))
    retrieved_metadata.append({"title": record["title"]})

# Convert list of vectors to a single array
retrieved_vectors_array = np.array(retrieved_vectors)
import_time = time.time() - start_time
print(f"[2.2] Vectors imported from database in {import_time:.4f} seconds")

# Initialize a new service with the imported vectors
reimport_service = RankingService(
    embedding_model=embedding_model,  # Original embedding model for new queries
    index=FaissIndex()
)

# Store the retrieved vectors
start_time = time.time()
reimport_service.storage.store_vectors(
    ids=retrieved_ids,
    vectors=retrieved_vectors_array,
    metadata=retrieved_metadata
)

# Build the index from the stored vectors
ids, vectors, _ = reimport_service.storage.get_vectors()
reimport_service.index.build(vectors, ids)
reimport_service.is_initialized = True
index_build_time = time.time() - start_time
print(f"[2.3] Index rebuilt from imported vectors in {index_build_time:.4f} seconds")

# Rank against user profile
start_time = time.time()
matches = reimport_service.find_similar(user_skills, k=len(listings))
ranking_time = time.time() - start_time
print(f"[2.4] Ranking completed in {ranking_time:.4f} seconds")

# Display results
print(f"\nTop matches for user with skills: {user_skills}\n")
for i, match in enumerate(matches):
    title = match.get('title', f"Job {match.get('id', i+1)}")
    print(f"{i+1}. {title} - Score: {match['score']:.4f}")

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Model loading time:    {model_load_time:.4f} seconds")
print(f"Embedding time:        {embedding_time:.4f} seconds")
print(f"Export time:           {export_time:.4f} seconds")
print(f"Import time:           {import_time:.4f} seconds")
print(f"Index rebuild time:    {index_build_time:.4f} seconds")
print(f"Ranking time:          {ranking_time:.4f} seconds")
print(f"Total rebuild+rank:    {index_build_time + ranking_time:.4f} seconds")