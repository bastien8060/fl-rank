# fl-rank

A fast, lightweight Python library for semantic tag matching and ranking.

## Why I Built This

> Context; I've been fascinated by ranking algorithms for content serving since a while. I wrote my first one at 12, a basic content feed where each item had used preset tags for indexing. The ranking used a simple JSC (Jaccard Similarity Coefficient) to find the most relevant posts for the user.

I hit this issue repeatedly when building tag-based matching systems, there was nothing lightweight that just *worked* for semantic matching. Every time I needed to rank items based on relevance to a user, I ended up rewriting the same vector similarity code.

FAISS is amazing, but using it directly means tons of boilerplate. Vector DBs are cool, but way overkill when I just want to rank a few thousand items. So I made `fl-rank` to solve my own headaches. It has one job: rank items based on semantic similarity between tags/keywords, and do it efficiently.

## Features

- Convert text/tags to vectors using Sentence Transformers
- Search with FAISS for fast (near O(n)) lookups (depends on the used similarity metric, ranking strategy, etc...)
- Support for different storage backends (memory, PostgreSQL/pgvector)
- Clean API that hides all the vector complexity 
- Persistence and import/export support
- No dependencies on specific web frameworks or databases
- Works well with 10-100k items
- Customizable preprocessing pipeline for tags
- Multiple similarity metrics (cosine, euclidean, dot product)
- Flexible ranking strategies
- Control over vector normalization

## Getting Started

### Install

```bash
pip install fl-rank

# For PostgreSQL/pgvector support
pip install fl-rank[pgvector]
```

### Basic Example

Here's a very simple example, ranks job listings against a user's skills:

```python
from fl_rank import RankingService
from fl_rank.embeddings import SentenceTransformerEmbedding
from fl_rank.indexes import FaissIndex

# Define job listings
listings = [
    {
        "id": "job1",
        "title": "Software Engineer",
        "skills": ["python", "nodejs", "react", "frontend", "github"]
    },
    {
        "id": "job2", 
        "title": "Data Scientist",
        "skills": ["python", "statistics", "machine learning", "pandas"]
    },
    {
        "id": "job3",
        "title": "DevOps Engineer",
        "skills": ["aws", "kubernetes", "docker", "linux", "ci/cd"]
    }
]

# Define user skills
user_skills = ["Python3.7", "NodeJS", "ReactJS", "ML/AI", "algorithms", "git"]

# Create ranking service
service = RankingService(
    embedding_model=SentenceTransformerEmbedding(),
    index=FaissIndex()
)

# Add job listings
service.add_items(listings, id_field="id", content_field="skills", metadata_fields=["title"])

# Find matches for user
matches = service.find_similar(user_skills, k=len(listings))

# Print results
for match in matches:
    print(f"{match['title']} - Score: {match['score']:.4f}")
```

### Advanced Example

Here's how to use some of the advanced features:

```python
from fl_rank import RankingService
from fl_rank.embeddings import SentenceTransformerEmbedding
from fl_rank.indexes import FaissIndex
from fl_rank.preprocessing import PreprocessingPipeline, CompoundTagTokenizer, VersionStripNormalizer
from fl_rank.ranking import DefaultRanker, EuclideanDistance, WeightedRankingStrategy

# Custom preprocessing pipeline
preprocessor = PreprocessingPipeline(
    tokenizer=CompoundTagTokenizer(separator="/"),
    normalizer=VersionStripNormalizer(add_base_form=True)
)

# Custom ranking strategy
def weight_by_importance(metadata):
    return metadata.get("importance", 5) / 10.0

weighted_strategy = WeightedRankingStrategy(
    metric=EuclideanDistance(),
    weight_fn=weight_by_importance
)

ranker = DefaultRanker()
ranker.strategy = weighted_strategy

# Create service with custom components
service = RankingService(
    embedding_model=SentenceTransformerEmbedding(),
    index=FaissIndex(),
    preprocessor=preprocessor,
    ranker=ranker
)

# Use as before...
```

## Integration with Databases

fl-rank works well with vector databases like PostgreSQL/pgvector. Here's a simple example that stores vectors and retrieves them:

```python
# Export vectors to database
ids, vectors, metadata = service.storage.get_vectors()

# Later, import them back
import numpy as np
from fl_rank import RankingService
from fl_rank.indexes import FaissIndex

# Create a new service
reimport_service = RankingService(
    embedding_model=embedding_model,
    index=FaissIndex()
)

# Store the retrieved vectors
reimport_service.storage.store_vectors(
    ids=retrieved_ids,
    vectors=np.array(retrieved_vectors),
    metadata=retrieved_metadata
)

# Build the index
reimport_service.index.build(vectors, ids)
```

## Components

fl-rank is modular, so you can swap components as needed:

### Embedding Models

```python
# Default sentence transformer
from fl_rank.embeddings import SentenceTransformerEmbedding
embedding = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

# For pre-computed vectors
from fl_rank.embeddings import PreprocessedVectorEmbedding
embedding = PreprocessedVectorEmbedding(dimension=384)
```

### Indexes

```python
# Simple flat index (fastest for <10k items)
from fl_rank.indexes import FaissIndex
index = FaissIndex()

# IVF index (better for larger datasets)
from fl_rank.indexes import FaissIndex, IndexConfig
index = FaissIndex(config=IndexConfig(index_type="ivf", nlist=100))
```

### Storage

```python
# In-memory storage (default)
from fl_rank.storage import InMemoryStorage
storage = InMemoryStorage()

# PostgreSQL/pgvector storage
from fl_rank.storage import PgVectorStorage, StorageConfig
storage = PgVectorStorage(
    config=StorageConfig(
        connection_string="postgresql://user:pass@localhost/db",
        table_name="vectors"
    )
)
```

### Preprocessing

```python
# Default preprocessing pipeline
from fl_rank.preprocessing import PreprocessingPipeline
preprocessor = PreprocessingPipeline()

# Custom components
from fl_rank.preprocessing import (
    CompoundTagTokenizer, LowercaseNormalizer,
    VersionStripNormalizer, StopwordFilter
)

# Build custom pipeline
custom_preprocessor = PreprocessingPipeline(
    tokenizer=CompoundTagTokenizer(separator="/"),
    normalizer=VersionStripNormalizer(add_base_form=True),
    filter_=StopwordFilter(stopwords={"and", "or", "the"})
)
```

### Ranking

```python
# Default similarity metric
from fl_rank.ranking import CosineSimilarity
metric = CosineSimilarity()

# Alternative metrics
from fl_rank.ranking import EuclideanDistance, DotProduct
euclidean = EuclideanDistance()
dot_product = DotProduct()

# Custom ranking strategies
from fl_rank.ranking import WeightedRankingStrategy, ReRankingStrategy
weighted = WeightedRankingStrategy(
    metric=CosineSimilarity(),
    weight_fn=lambda metadata: metadata.get("importance", 5) / 10.0
)

two_phase = ReRankingStrategy(
    primary_metric=EuclideanDistance(),
    secondary_metric=CosineSimilarity(),
    prefilter_k=100
)
```

## Performance

fl-rank is designed to be fast, even for medium-sized datasets:
- Index building: milliseconds for small datasets (~5K items)
- Query time: 15-20ms for ranking against full dataset
- Scales nicely to ~100K items before needing more optimization

The main performance hit is usually loading the embedding model (~3s), but that's a one-time cost at startup.

## Roadmap

Some things I'm planning to add:
- [ ] Add support for Indexing caching / saving
- [ ] More embedding models (OpenAI, Cohere, etc)
- [ ] Batched operations for better throughput
- [ ] Support for more vector DBs (Pinecone, Milvus, etc)
- [ ] Better documentation and examples
- [ ] CLI tools

## Contributing

Contributions are welcome! Feel free to open issues or PRs if you have ideas for improvements.

## License

MIT License - see [LICENSE](LICENSE) for details.