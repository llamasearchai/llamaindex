#!/usr/bin/env python3
"""
Distributed indexing example for LlamaIndex.

This example demonstrates how to distribute indexing and querying across multiple
processes or machines. It shows:
1. Creating multiple index shards
2. Distributing documents across shards
3. Querying across all shards and merging results
"""

import logging
import multiprocessing
import os
import sys
import time
import uuid
from typing import Dict, List, Any, Callable, Optional
import random

# Add the src directory to the path so we can import llamaindex
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import llamaindex as li

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_embedding(text: str) -> List[float]:
    """
    Generate a simple embedding for demonstration purposes.
    In a real application, you would use a proper embedding model.
    """
    # This is just a simple hash-based embedding for demonstration
    # Do not use this in production!
    hash_val = hash(text)
    np.random.seed(hash_val)
    return list(np.random.rand(128))


def create_sample_documents(count: int) -> List[li.Document]:
    """
    Create sample documents for testing.
    
    Args:
        count: Number of documents to create
        
    Returns:
        List of sample documents
    """
    topics = ["machine learning", "artificial intelligence", "data science", 
              "natural language processing", "computer vision", "database systems",
              "distributed systems", "cloud computing", "web development", 
              "cybersecurity", "blockchain", "quantum computing"]
    
    documents = []
    for i in range(count):
        topic = random.choice(topics)
        content = f"This is document {i} about {topic}. It contains information about various aspects of {topic}."
        metadata = {
            "topic": topic,
            "doc_number": i,
            "timestamp": time.time()
        }
        
        doc = li.Document(
            content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def index_shard_worker(
    shard_id: int, 
    documents: List[li.Document],
    index_type: str,
    index_dir: str,
    embedding_function: Optional[Callable] = None,
) -> None:
    """
    Worker function to create and populate an index shard.
    
    Args:
        shard_id: ID of this shard
        documents: Documents to index
        index_type: Type of index to create
        index_dir: Directory to save the index
        embedding_function: Function to generate embeddings (for vector/hybrid indexes)
    """
    # Configure logging for this process
    logger = logging.getLogger(f"shard-{shard_id}")
    logger.info(f"Starting indexing for shard {shard_id} with {len(documents)} documents")
    
    # Create the index
    index_builder = li.IndexBuilder()
    index_builder.with_id(f"shard-{shard_id}")
    index_builder.with_name(f"Shard {shard_id}")
    index_builder.with_index_type(index_type)
    
    if embedding_function and index_type in ["vector", "hybrid"]:
        index_builder.with_embedding_function(embedding_function)
    
    # Build the index
    index = index_builder.build()
    
    # Add the documents
    start_time = time.time()
    index.add_documents(documents)
    end_time = time.time()
    
    logger.info(f"Indexed {len(documents)} documents in {end_time - start_time:.2f} seconds")
    
    # Save the index
    shard_dir = os.path.join(index_dir, f"shard-{shard_id}")
    os.makedirs(shard_dir, exist_ok=True)
    index.persist(shard_dir)
    
    logger.info(f"Shard {shard_id} saved to {shard_dir}")


def query_shard_worker(
    shard_id: int,
    query_text: str,
    index_dir: str,
    index_type: str,
    search_type: str,
    top_k: int,
    embedding_function: Optional[Callable] = None,
    result_queue: multiprocessing.Queue = None,
) -> None:
    """
    Worker function to query an index shard.
    
    Args:
        shard_id: ID of the shard to query
        query_text: Query text
        index_dir: Directory where indexes are stored
        index_type: Type of index (inverted, vector, hybrid)
        search_type: Type of search (keyword, semantic, hybrid)
        top_k: Number of results to return
        embedding_function: Function to generate embeddings
        result_queue: Queue to put results in
    """
    logger = logging.getLogger(f"query-shard-{shard_id}")
    logger.info(f"Querying shard {shard_id}")
    
    # Load the index
    shard_dir = os.path.join(index_dir, f"shard-{shard_id}")
    
    try:
        if index_type == "inverted":
            index = li.InvertedIndex.load(shard_dir)
        elif index_type == "vector":
            index = li.VectorIndex.load(shard_dir)
            if embedding_function:
                index.set_embedding_function(embedding_function)
        else:  # hybrid
            index = li.HybridIndex.load(shard_dir)
            if embedding_function:
                index.set_embedding_function(embedding_function)
    except Exception as e:
        logger.error(f"Failed to load shard {shard_id}: {e}")
        if result_queue:
            result_queue.put((shard_id, None))
        return
    
    # Create a query processor
    processor = li.QueryProcessor(index=index)
    if embedding_function and (search_type in ["semantic", "hybrid"] or index_type in ["vector", "hybrid"]):
        processor.embedding_function = embedding_function
    
    # Execute the query
    try:
        start_time = time.time()
        result = processor.query(
            query_text=query_text,
            top_k=top_k,
            search_type=search_type
        )
        end_time = time.time()
        
        logger.info(f"Query executed in {(end_time - start_time) * 1000:.2f}ms, found {len(result.matches)} matches")
        
        # Put results in the queue if provided
        if result_queue:
            result_queue.put((shard_id, result))
        
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        if result_queue:
            result_queue.put((shard_id, None))
        return None


def merge_results(
    results: Dict[int, li.QueryResult],
    query_text: str,
    top_k: int,
) -> li.QueryResult:
    """
    Merge results from multiple shards.
    
    Args:
        results: Dictionary mapping shard IDs to QueryResults
        query_text: The original query text
        top_k: Number of top results to keep
        
    Returns:
        Merged QueryResult
    """
    logger.info(f"Merging results from {len(results)} shards")
    
    # Create a new result
    merged_result = li.QueryResult(
        query_text=query_text,
        search_type="distributed"
    )
    
    # Collect all matches
    all_matches = []
    for shard_id, result in results.items():
        if result and result.matches:
            for match in result.matches:
                # Mark which shard this came from
                match.metadata["shard_id"] = shard_id
                all_matches.append(match)
    
    # Sort all matches by score
    all_matches.sort(key=lambda x: x.score, reverse=True)
    
    # Take top_k
    top_matches = all_matches[:top_k]
    
    # Add to the result
    for match in top_matches:
        merged_result.add_match(match)
    
    logger.info(f"Merged {len(all_matches)} matches into {len(merged_result.matches)} top results")
    
    return merged_result


def distributed_indexing_example(
    num_shards: int = 3,
    docs_per_shard: int = 20,
    index_type: str = "hybrid",
):
    """
    Run the distributed indexing example.
    
    Args:
        num_shards: Number of index shards to create
        docs_per_shard: Number of documents per shard
        index_type: Type of index to create (inverted, vector, hybrid)
    """
    logger.info(f"Starting distributed indexing example with {num_shards} shards")
    
    # Create a temporary directory for the indexes
    import tempfile
    import shutil
    
    index_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory: {index_dir}")
    
    try:
        # Create sample documents for each shard
        all_documents = []
        for i in range(num_shards):
            shard_docs = create_sample_documents(docs_per_shard)
            all_documents.extend(shard_docs)
            
            # Start a process to index this shard
            p = multiprocessing.Process(
                target=index_shard_worker,
                args=(i, shard_docs, index_type, index_dir, generate_embedding)
            )
            p.start()
            p.join()
        
        logger.info(f"Indexed {len(all_documents)} documents across {num_shards} shards")
        
        # Now query the shards
        sample_queries = [
            "machine learning algorithms",
            "distributed systems architecture",
            "natural language processing techniques",
            "computer vision applications"
        ]
        
        for query_text in sample_queries:
            logger.info(f"\nExecuting distributed query: '{query_text}'")
            
            # Create a queue for results
            result_queue = multiprocessing.Queue()
            
            # Start a process to query each shard
            processes = []
            for i in range(num_shards):
                p = multiprocessing.Process(
                    target=query_shard_worker,
                    args=(
                        i, query_text, index_dir, index_type, 
                        "hybrid", 5, generate_embedding, result_queue
                    )
                )
                processes.append(p)
                p.start()
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
            
            # Collect results
            results = {}
            while not result_queue.empty():
                shard_id, result = result_queue.get()
                if result:
                    results[shard_id] = result
            
            # Merge results
            if results:
                merged_result = merge_results(results, query_text, 5)
                
                # Display top results
                logger.info(f"Top results for query '{query_text}':")
                for i, match in enumerate(merged_result.matches):
                    logger.info(f"  {i+1}. [Shard {match.metadata['shard_id']}] Score: {match.score:.3f}")
                    logger.info(f"     Content: {match.content[:100]}...")
                    logger.info(f"     Topic: {match.metadata.get('topic', 'N/A')}")
                    logger.info("")
            else:
                logger.info("No results found across any shards")
    
    finally:
        # Clean up
        logger.info(f"Cleaning up temporary directory: {index_dir}")
        shutil.rmtree(index_dir)


def distributed_index_creation():
    """Example showing how to create a distributed index."""
    logger.info("=== Distributed Index Creation ===")
    distributed_indexing_example(
        num_shards=3,
        docs_per_shard=20,
        index_type="hybrid"
    )


def main():
    """Run all the examples."""
    logger.info("Starting LlamaIndex distributed examples")
    
    distributed_index_creation()
    
    logger.info("\nExamples completed")


if __name__ == "__main__":
    main() 