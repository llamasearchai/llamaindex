"""
Setup script for the LlamaIndex package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamaindex",
    version="0.1.0",
    author="LlamaSearch.ai",
    author_email="info@llamasearch.ai",
    description="A high-performance indexing and search library for LlamaSearch.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearch/llamaindex",
    project_urls={
        "Bug Tracker": "https://github.com/llamasearch/llamaindex/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pydantic>=1.9.0",
    ],
    extras_require={
        "mlx": ["mlx>=0.0.1", "transformers>=4.30.0"],
        "sentence-transformers": ["sentence-transformers>=2.2.0"],
        "faiss": ["faiss-cpu>=1.7.0"],
        "hnsw": ["hnswlib>=0.7.0"],
        "annoy": ["annoy>=1.17.0"],
        "dev": [
            "black",
            "isort",
            "mypy",
            "pytest",
            "pytest-cov",
        ],
        "all": [
            "mlx>=0.0.1",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "hnswlib>=0.7.0",
            "annoy>=1.17.0",
        ],
    },
) 