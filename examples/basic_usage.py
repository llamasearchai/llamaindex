"""
Basic usage example for LlamaIndex
"""
from llamaindex import Index, Document, Filter
from llamaindex.embeddings import MLXEmbeddings

def main():
    """Main function demonstrating basic LlamaIndex usage."""
    print("Initializing LlamaIndex...")
    
    # Create an index with default settings and MLX embeddings (for Apple Silicon)
    try:
        index = Index(embeddings=MLXEmbeddings())
        print("Using MLX embeddings on Apple Silicon")
    except ImportError:
        # Fall back to SentenceTransformer embeddings if MLX not available
        from llamaindex.embeddings import SentenceTransformerEmbeddings
        index = Index(embeddings=SentenceTransformerEmbeddings())
        print("Using SentenceTransformer embeddings")
    
    # Add documents
    print("\nAdding documents to the index...")
    documents = [
        Document(
            text="LlamaSearch is a new search engine for developers.",
            metadata={"source": "website", "category": "technology", "date": "2023-05-01"}
        ),
        Document(
            text="Vector search enables semantic search capabilities.",
            metadata={"source": "blog", "category": "technology", "date": "2023-05-15"}
        ),
        Document(
            text="MLX is an array framework for Apple Silicon.",
            metadata={"source": "documentation", "category": "technology", "date": "2023-06-01"}
        ),
        Document(
            text="Python is a popular programming language for data science and machine learning.",
            metadata={"source": "article", "category": "programming", "date": "2023-04-20"}
        ),
        Document(
            text="The new MacBook Pro features Apple's M2 Pro chip.",
            metadata={"source": "news", "category": "hardware", "date": "2023-07-01"}
        ),
    ]
    
    doc_ids = index.add_documents(documents)
    print(f"Added {len(doc_ids)} documents to the index")
    
    # Basic search
    print("\nPerforming basic search...")
    results = index.search("semantic search technology")
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.4f} - {result.document.text}")
    
    # Search with filters
    print("\nPerforming search with filters...")
    filtered_results = index.search(
        "technology",
        filters=Filter.and_(
            Filter.eq("category", "technology"),
            Filter.gte("date", "2023-06-01")
        )
    )
    
    print(f"Found {len(filtered_results)} filtered results:")
    for i, result in enumerate(filtered_results):
        print(f"{i+1}. Score: {result.score:.4f} - {result.document.text}")
        print(f"   Metadata: {result.document.metadata}")
    
    # Hybrid search
    print("\nPerforming hybrid search...")
    hybrid_results = index.search(
        "programming language",
        hybrid_search=True,
        hybrid_alpha=0.7  # Weight between vector (1.0) and keyword (0.0) search
    )
    
    print(f"Found {len(hybrid_results)} hybrid results:")
    for i, result in enumerate(hybrid_results):
        print(f"{i+1}. Score: {result.score:.4f} - {result.document.text}")
    
    # Save the index
    print("\nSaving index...")
    save_path = index.save("./basic_index")
    print(f"Index saved to {save_path}")
    
    # Load the index
    print("\nLoading index...")
    loaded_index = Index.load(save_path)
    print(f"Loaded index with {len(loaded_index.documents)} documents")
    
    # Search the loaded index
    print("\nSearching loaded index...")
    loaded_results = loaded_index.search("Apple Silicon")
    
    print(f"Found {len(loaded_results)} results in loaded index:")
    for i, result in enumerate(loaded_results):
        print(f"{i+1}. Score: {result.score:.4f} - {result.document.text}")

if __name__ == "__main__":
    main() 