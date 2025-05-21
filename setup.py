# setup.py
from setuptools import setup, find_packages

setup(
    name="fl-rank",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
    ],
    extras_require={
        "pgvector": ["psycopg2-binary>=2.9.0", "sqlalchemy>=1.4.0"],
        "dev": ["pytest>=6.0.0", "black>=22.0.0", "isort>=5.0.0"],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Fast Light Ranking System for semantic vector operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fl-rank",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

# pyproject.toml
"""
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ['py37', 'py38', 'py39']

[tool.isort]
profile = "black"
"""

# README.md
"""
# fl-rank: Fast Light Vector Ranking System

A modular, high-performance vector ranking system designed for semantic search and recommendation applications.

## Features

- Fast vector similarity search with FAISS
- Modular architecture for different embedding models and storage backends
- Support for PostgreSQL/pgvector, in-memory, and custom storage
- Service-oriented design for easy integration
- Batch processing capabilities

## Installation

```bash
pip install fl-rank
```

For pgvector support:
```bash
pip install fl-rank[pgvector]
```

## Quick Start

```python
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

# Add items
items = [
    {"id": "1", "tags": ["python", "programming", "web development"]},
    {"id": "2", "tags": ["data science", "machine learning", "python"]},
    {"id": "3", "tags": ["javascript", "frontend", "web development"]},
]

# Process items
service.add_items(items, id_field="id", content_field="tags")

# Find similar items
query = ["python", "machine learning"]
results = service.find_similar(query, k=2)
print(results)
```

## Documentation

For detailed documentation and examples, see the [GitHub repository](https://github.com/yourusername/fl-rank).
"""

# LICENSE
"""
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""