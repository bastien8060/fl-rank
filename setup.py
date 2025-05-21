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