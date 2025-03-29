"""
Advanced usage examples for LlamaIndex
"""
import os
import numpy as np
from llamaindex import Index, Document, Filter, IndexConfig
from llamaindex.embeddings import SentenceTransformerEmbeddings, CustomEmbeddings

def custom_embedding_example():
    """Example using custom embeddings."""
    print("\n=== Custom Embeddings Example ===")
    
    # Define a simple embedding function that uses random vectors
    # In a real application, this could be any custom embedding model
    def random_embedding_function(texts):
        """Generate random embeddings for demonstration purposes."""
        return [np.random.randn(128).tolist() for _ in texts]
    
    # Create custom embeddings
    embeddings = CustomEmbeddings(
        embed_function=random_embedding_function,
        dimensions=128
    )
    
    # Create index with custom embeddings
    index = Index(embeddings=embeddings)
    
    # Add some documents
    documents = [
        Document(
            text="This is document one about custom embeddings.",
            metadata={"type": "example", "id": 1}
        ),
        Document(
            text="This is document two about vector search.",
            metadata={"type": "example", "id": 2}
        ),
        Document(
            text="This is document three about machine learning.",
            metadata={"type": "tutorial", "id": 3}
        ),
    ]
    
    index.add_documents(documents)
    
    # Search
    results = index.search("custom embeddings")
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.4f} - {result.document.text}")
    
    return index

def advanced_filtering_example():
    """Example demonstrating advanced filtering capabilities."""
    print("\n=== Advanced Filtering Example ===")
    
    # Create index with sentence transformer embeddings
    index = Index(embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
    
    # Add documents with rich metadata
    documents = [
        Document(
            text="Python is a popular programming language for data science and AI.",
            metadata={
                "language": "python",
                "type": "language",
                "tags": ["programming", "data-science", "ai"],
                "difficulty": "beginner",
                "year": 2023,
                "author": "John Doe"
            }
        ),
        Document(
            text="JavaScript is widely used for web development and runs in browsers.",
            metadata={
                "language": "javascript",
                "type": "language",
                "tags": ["programming", "web", "frontend"],
                "difficulty": "intermediate",
                "year": 2022,
                "author": "Jane Smith"
            }
        ),
        Document(
            text="TensorFlow is a machine learning framework from Google.",
            metadata={
                "language": "python",
                "type": "framework",
                "tags": ["ai", "machine-learning", "deep-learning"],
                "difficulty": "advanced",
                "year": 2023,
                "author": "Google"
            }
        ),
        Document(
            text="React is a JavaScript library for building user interfaces.",
            metadata={
                "language": "javascript",
                "type": "library",
                "tags": ["web", "frontend", "ui"],
                "difficulty": "intermediate",
                "year": 2021,
                "author": "Facebook"
            }
        ),
        Document(
            text="PyTorch is a machine learning framework from Facebook.",
            metadata={
                "language": "python",
                "type": "framework",
                "tags": ["ai", "machine-learning", "deep-learning"],
                "difficulty": "advanced",
                "year": 2023,
                "author": "Facebook"
            }
        ),
    ]
    
    index.add_documents(documents)
    
    # Example 1: Simple filter
    print("\nFilter by language == 'python':")
    results = index.search(
        "machine learning",
        filters=Filter.eq("language", "python")
    )
    for result in results:
        print(f"Score: {result.score:.4f} - {result.document.text}")
    
    # Example 2: Complex AND filter
    print("\nFilter by (language == 'python' AND type == 'framework'):")
    results = index.search(
        "machine learning",
        filters=Filter.and_(
            Filter.eq("language", "python"),
            Filter.eq("type", "framework")
        )
    )
    for result in results:
        print(f"Score: {result.score:.4f} - {result.document.text}")
    
    # Example 3: Complex OR filter
    print("\nFilter by (difficulty == 'advanced' OR year >= 2023):")
    results = index.search(
        "programming",
        filters=Filter.or_(
            Filter.eq("difficulty", "advanced"),
            Filter.gte("year", 2023)
        )
    )
    for result in results:
        print(f"Score: {result.score:.4f} - {result.document.text}")
        print(f"   Metadata: difficulty={result.document.metadata['difficulty']}, year={result.document.metadata['year']}")
    
    # Example 4: IN filter
    print("\nFilter by 'ai' in tags:")
    results = index.search(
        "framework",
        filters=Filter.in_("tags", "ai")
    )
    for result in results:
        print(f"Score: {result.score:.4f} - {result.document.text}")
        print(f"   Tags: {result.document.metadata['tags']}")
    
    # Example 5: NOT filter
    print("\nFilter by NOT (author == 'Facebook'):")
    results = index.search(
        "programming",
        filters=Filter.not_(Filter.eq("author", "Facebook"))
    )
    for result in results:
        print(f"Score: {result.score:.4f} - {result.document.text}")
        print(f"   Author: {result.document.metadata['author']}")
    
    return index

def custom_config_example():
    """Example demonstrating custom index configuration."""
    print("\n=== Custom Index Configuration Example ===")
    
    # Create a custom configuration
    config = IndexConfig(
        vector_store_type="flat",        # Use flat vector store (numpy-based)
        distance_metric="cosine",        # Use cosine similarity
        hybrid_search=True,              # Enable hybrid search by default
        hybrid_alpha=0.7,                # Set hybrid search weight to 0.7 (more vector, less keyword)
        top_k=5,                         # Return top 5 results by default
        chunk_size=256,                  # Use smaller chunks for better precision
        chunk_overlap=64,                # Use smaller overlap between chunks
        embedding_model="sentence-transformers",  # Use sentence-transformers embeddings
    )
    
    # Create index with custom config
    index = Index(
        config=config,
        embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    
    # Add some documents
    documents = []
    for i in range(10):
        documents.append(Document(
            text=f"This is test document {i+1} with some random content for testing hybrid search and configuration.",
            metadata={"doc_id": i+1, "test": True}
        ))
    
    index.add_documents(documents)
    
    # Search with default config settings (should use hybrid search)
    results = index.search("test document")
    
    print(f"Found {len(results)} results with custom config:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.4f} - {result.document.text}")
    
    return index

def main():
    """Run all advanced examples."""
    # Example 1: Custom embeddings
    custom_index = custom_embedding_example()
    
    # Example 2: Advanced filtering
    filter_index = advanced_filtering_example()
    
    # Example 3: Custom configuration
    config_index = custom_config_example()
    
    # Save an index as an example
    print("\n=== Saving and Loading Index ===")
    save_dir = "./advanced_index"
    filter_index.save(save_dir)
    print(f"Index saved to {save_dir}")
    
    # Load the index
    loaded_index = Index.load(save_dir)
    print(f"Loaded index with {len(loaded_index.documents)} documents")
    
    # Verify it works
    test_results = loaded_index.search("python", top_k=2)
    print(f"Test search results: {len(test_results)} results")
    for result in test_results:
        print(f"Score: {result.score:.4f} - {result.document.text}")

if __name__ == "__main__":
    main() 