# LlamaIndex

A high-performance indexing and search library for LlamaSearch.ai, optimized for semantic search with vector embeddings.

## Features

- **Vector Indexing**: Fast vector search with multiple backend options (HNSW, FAISS, flat)
- **Hybrid Search**: Combine vector search with keyword search for better results
- **Apple Silicon Optimization**: Built-in MLX support for state-of-the-art performance on Apple Silicon
- **Easy API**: Simple, intuitive interface for indexing and searching documents
- **Metadata Filtering**: Filter search results based on document metadata
- **Flexible Embeddings**: Support for multiple embedding models and custom embeddings

## Installation

```bash
# Base installation
pip install llamaindex

# With MLX support for Apple Silicon
pip install llamaindex[mlx]

# With all dependencies
pip install llamaindex[all]

# With specific backends
pip install llamaindex[sentence-transformers,hnsw]
```

## Quick Start

```python
from llamaindex import Index, Document, Filter
from llamaindex.embeddings import SentenceTransformerEmbeddings

# Create an index with SentenceTransformer embeddings
index = Index(embeddings=SentenceTransformerEmbeddings())

# Add documents
documents = [
    Document(
        text="LlamaSearch is a new search engine for developers.",
        metadata={"source": "website", "category": "technology"}
    ),
    Document(
        text="Vector search enables semantic search capabilities.",
        metadata={"source": "blog", "category": "technology"}
    ),
    Document(
        text="Python is a popular programming language for data science.",
        metadata={"source": "article", "category": "programming"}
    ),
]

index.add_documents(documents)

# Basic search
results = index.search("semantic search")
for result in results:
    print(f"Score: {result.score:.4f} - {result.document.text}")

# Search with filters
filtered_results = index.search(
    "technology",
    filters=Filter.eq("category", "technology")
)

# Save and load index
index.save("./my_index")
loaded_index = Index.load("./my_index")
```

## Advanced Features

### Hybrid Search

Combine vector and keyword search for better results, especially for rare terms:

```python
results = index.search(
    "programming language",
    hybrid_search=True,
    hybrid_alpha=0.7  # Weight between vector (1.0) and keyword (0.0) search
)
```

### Custom Embeddings

Use your own embedding function:

```python
from llamaindex.embeddings import CustomEmbeddings

def my_embed_function(texts):
    # Your custom embedding logic here
    return [[0.1, 0.2, 0.3] for _ in texts]

index = Index(embeddings=CustomEmbeddings(
    embed_function=my_embed_function,
    dimensions=3
))
```

### Apple Silicon Optimization

MLX embeddings for Apple Silicon:

```python
from llamaindex.embeddings import MLXEmbeddings

# Use MLX for optimized performance on Apple Silicon
index = Index(embeddings=MLXEmbeddings())
```

## Documentation

For full documentation, see [docs.llamasearch.ai/llamaindex](https://docs.llamasearch.ai/llamaindex).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 