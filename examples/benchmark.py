#!/usr/bin/env python3
"""
Benchmark script for LlamaIndex.

This script benchmarks different index types and search strategies, measuring
performance metrics like indexing time, query latency, and result quality.
"""

import logging
import os
import sys
import time
import json
from typing import Dict, List, Any, Callable, Optional, Tuple
import random
import math
from datetime import datetime

# Add the src directory to the path so we can import llamaindex
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
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


def create_test_documents(count: int, vocab_size: int = 5000) -> List[li.Document]:
    """
    Create test documents with controlled vocabulary.
    
    Args:
        count: Number of documents to create
        vocab_size: Size of vocabulary to use
        
    Returns:
        List of documents
    """
    # Generate a vocabulary of random words
    np.random.seed(42)  # For reproducibility
    vocab = [f"word{i}" for i in range(vocab_size)]
    
    documents = []
    for i in range(count):
        # Generate a random document with 20-50 words
        doc_length = random.randint(20, 50)
        words = [random.choice(vocab) for _ in range(doc_length)]
        content = " ".join(words)
        
        # Add some metadata
        metadata = {
            "doc_id": i,
            "length": doc_length,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        doc = li.Document(content=content, metadata=metadata)
        documents.append(doc)
    
    return documents


def create_test_queries(count: int, vocab_size: int = 5000) -> List[str]:
    """
    Create test queries with controlled vocabulary.
    
    Args:
        count: Number of queries to create
        vocab_size: Size of vocabulary to use
        
    Returns:
        List of query strings
    """
    # Use the same seed as document creation for consistency
    np.random.seed(42)
    vocab = [f"word{i}" for i in range(vocab_size)]
    
    queries = []
    for _ in range(count):
        # Generate queries with 1-3 words
        query_length = random.randint(1, 3)
        words = [random.choice(vocab) for _ in range(query_length)]
        query = " ".join(words)
        queries.append(query)
    
    return queries


def benchmark_indexing(
    documents: List[li.Document],
    index_types: List[str],
    embedding_function: Optional[Callable] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark document indexing performance.
    
    Args:
        documents: Documents to index
        index_types: List of index types to benchmark
        embedding_function: Function to generate embeddings
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking indexing performance with {len(documents)} documents")
    
    results = {}
    
    # Create indexes of each type
    for index_type in index_types:
        logger.info(f"Testing {index_type} index...")
        
        # Create the index
        builder = li.IndexBuilder().with_index_type(index_type)
        if embedding_function and index_type in ["vector", "hybrid"]:
            builder.with_embedding_function(embedding_function)
        index = builder.build()
        
        # Measure indexing time
        start_time = time.time()
        index.add_documents(documents)
        end_time = time.time()
        
        indexing_time = end_time - start_time
        docs_per_second = len(documents) / indexing_time
        
        # Collect memory usage - this is approximate
        import psutil
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        results[index_type] = {
            "indexing_time_seconds": indexing_time,
            "docs_per_second": docs_per_second,
            "memory_usage_mb": memory_usage,
        }
        
        logger.info(f"  Indexed {len(documents)} documents in {indexing_time:.2f} seconds")
        logger.info(f"  Throughput: {docs_per_second:.2f} docs/second")
        logger.info(f"  Memory usage: {memory_usage:.2f} MB")
    
    return results


def benchmark_querying(
    documents: List[li.Document],
    queries: List[str],
    index_types: List[str],
    search_types: List[str],
    embedding_function: Optional[Callable] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Benchmark query performance.
    
    Args:
        documents: Documents to index
        queries: Queries to execute
        index_types: List of index types to benchmark
        search_types: List of search types to benchmark
        embedding_function: Function to generate embeddings
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info(f"Benchmarking query performance with {len(documents)} documents and {len(queries)} queries")
    
    results = {}
    
    # Create indexes of each type
    for index_type in index_types:
        results[index_type] = {}
        
        # Create the index
        builder = li.IndexBuilder().with_index_type(index_type)
        if embedding_function and index_type in ["vector", "hybrid"]:
            builder.with_embedding_function(embedding_function)
        index = builder.build()
        
        # Add documents
        logger.info(f"Building {index_type} index...")
        index.add_documents(documents)
        
        # Create query processor
        processor = li.QueryProcessor(index=index)
        if embedding_function:
            processor.embedding_function = embedding_function
        
        # Test each search type
        for search_type in search_types:
            # Skip incompatible combinations
            if (search_type == "semantic" and index_type == "inverted") or \
               (search_type == "keyword" and index_type == "vector"):
                logger.info(f"Skipping incompatible combination: {index_type} index with {search_type} search")
                continue
                
            logger.info(f"Testing {index_type} index with {search_type} search...")
            
            # Execute queries
            query_times = []
            for query in queries:
                start_time = time.time()
                processor.query(query_text=query, search_type=search_type)
                end_time = time.time()
                query_time_ms = (end_time - start_time) * 1000
                query_times.append(query_time_ms)
            
            # Calculate statistics
            avg_query_time = sum(query_times) / len(query_times)
            p95_query_time = sorted(query_times)[int(len(query_times) * 0.95)]
            
            results[index_type][search_type] = {
                "avg_query_time_ms": avg_query_time,
                "p95_query_time_ms": p95_query_time,
                "queries_per_second": 1000 / avg_query_time,
            }
            
            logger.info(f"  Average query time: {avg_query_time:.2f} ms")
            logger.info(f"  P95 query time: {p95_query_time:.2f} ms")
            logger.info(f"  Throughput: {1000 / avg_query_time:.2f} queries/second")
    
    return results


def plot_benchmark_results(
    indexing_results: Dict[str, Dict[str, float]],
    querying_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
):
    """
    Generate plots from benchmark results.
    
    Args:
        indexing_results: Results from indexing benchmark
        querying_results: Results from querying benchmark
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot indexing times
    plt.figure(figsize=(10, 6))
    index_types = list(indexing_results.keys())
    indexing_times = [indexing_results[idx]["indexing_time_seconds"] for idx in index_types]
    
    plt.barh(index_types, indexing_times, color='skyblue')
    plt.xlabel('Indexing Time (seconds)')
    plt.ylabel('Index Type')
    plt.title('Indexing Time by Index Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'indexing_times.png'))
    
    # Plot indexing throughput
    plt.figure(figsize=(10, 6))
    throughput = [indexing_results[idx]["docs_per_second"] for idx in index_types]
    
    plt.barh(index_types, throughput, color='lightgreen')
    plt.xlabel('Documents per Second')
    plt.ylabel('Index Type')
    plt.title('Indexing Throughput by Index Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'indexing_throughput.png'))
    
    # Plot query times
    plt.figure(figsize=(12, 8))
    
    # Collect data for plotting
    index_search_pairs = []
    query_times = []
    
    for index_type, search_results in querying_results.items():
        for search_type, metrics in search_results.items():
            index_search_pairs.append(f"{index_type}-{search_type}")
            query_times.append(metrics["avg_query_time_ms"])
    
    # Sort by query time
    sorted_data = sorted(zip(index_search_pairs, query_times), key=lambda x: x[1])
    index_search_pairs, query_times = zip(*sorted_data)
    
    plt.barh(index_search_pairs, query_times, color='salmon')
    plt.xlabel('Average Query Time (ms)')
    plt.ylabel('Index-Search Combination')
    plt.title('Query Performance by Index and Search Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'query_times.png'))
    
    # Plot query throughput
    plt.figure(figsize=(12, 8))
    query_throughput = []
    
    for index_type, search_results in querying_results.items():
        for search_type, metrics in search_results.items():
            if f"{index_type}-{search_type}" in index_search_pairs:
                query_throughput.append(metrics["queries_per_second"])
    
    plt.barh(index_search_pairs, query_throughput, color='lightblue')
    plt.xlabel('Queries per Second')
    plt.ylabel('Index-Search Combination')
    plt.title('Query Throughput by Index and Search Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'query_throughput.png'))


def run_benchmark(
    doc_count: int = 1000,
    query_count: int = 100,
    output_dir: str = "benchmark_results",
):
    """
    Run a complete benchmark of LlamaIndex.
    
    Args:
        doc_count: Number of documents to use
        query_count: Number of queries to execute
        output_dir: Directory to save results
    """
    logger.info(f"Starting LlamaIndex benchmark with {doc_count} documents and {query_count} queries")
    
    # Create test data
    documents = create_test_documents(doc_count)
    queries = create_test_queries(query_count)
    
    # Define index and search types to benchmark
    index_types = ["inverted", "vector", "hybrid"]
    search_types = ["keyword", "semantic", "hybrid"]
    
    # Run benchmarks
    indexing_results = benchmark_indexing(
        documents=documents,
        index_types=index_types,
        embedding_function=generate_embedding,
    )
    
    querying_results = benchmark_querying(
        documents=documents,
        queries=queries,
        index_types=index_types,
        search_types=search_types,
        embedding_function=generate_embedding,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump({
            "config": {
                "doc_count": doc_count,
                "query_count": query_count,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "indexing": indexing_results,
            "querying": querying_results,
        }, f, indent=2)
    
    # Generate plots
    try:
        plot_benchmark_results(indexing_results, querying_results, output_dir)
        logger.info(f"Plots saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
    
    logger.info("Benchmark completed")


def main():
    """Run the benchmark with default parameters."""
    output_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    run_benchmark(
        doc_count=500,   # Use a smaller number for quick testing
        query_count=50,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main() 