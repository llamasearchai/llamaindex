#!/usr/bin/env python3
"""
Web demo for LlamaIndex.

This is a simple Flask web application that demonstrates the capabilities
of LlamaIndex. It allows users to:
1. Upload documents to the index
2. Search the index using different search strategies
3. View search results and document details
"""

import os
import sys
import logging
import uuid
import tempfile
from typing import Dict, List, Optional, Any

# Add the src directory to the path so we can import llamaindex
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import llamaindex as li
from flask import Flask, request, jsonify, render_template, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder="web_demo_templates")

# Global variables
index = None
index_dir = os.path.join(tempfile.gettempdir(), "llamaindex_demo")
uploaded_files = {}


def generate_simple_embedding(text: str) -> List[float]:
    """
    Generate a simple embedding for demonstration purposes.
    In a real application, you would use a proper embedding model.
    """
    import numpy as np
    hash_val = hash(text)
    np.random.seed(hash_val)
    return list(np.random.rand(128))


def init_index():
    """Initialize the index."""
    global index
    
    # Create index directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    # Try to load existing index, or create a new one
    try:
        logger.info(f"Attempting to load index from {index_dir}")
        index = li.HybridIndex.load(index_dir)
        logger.info(f"Loaded index with {len(index.doc_ids)} documents")
    except Exception as e:
        logger.info(f"Could not load index, creating new one: {e}")
        index = li.HybridIndex(
            index_name="web_demo_index",
            embedding_function=generate_simple_embedding,
            hybrid_alpha=0.5  # Balance between keyword and semantic search
        )
        index.persist(index_dir)


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Upload a document to the index.
    
    Expects a JSON payload with:
    - content: document text content
    - metadata: optional metadata for the document
    """
    try:
        data = request.json
        
        if not data or 'content' not in data:
            return jsonify({'error': 'No content provided'}), 400
        
        # Create a document
        doc_id = str(uuid.uuid4())
        content = data['content']
        metadata = data.get('metadata', {})
        
        if 'title' in data:
            metadata['title'] = data['title']
        
        doc = li.Document(
            content=content,
            doc_id=doc_id,
            metadata=metadata
        )
        
        # Add to index
        index.add_document(doc)
        
        # Save the index
        index.persist(index_dir)
        
        # Keep track of the uploaded file
        uploaded_files[doc_id] = {
            'doc_id': doc_id,
            'content': content[:100] + "..." if len(content) > 100 else content,
            'metadata': metadata
        }
        
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'message': 'Document added to index'
        })
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Search the index.
    
    Expects a JSON payload with:
    - query: the search query string
    - search_type: one of "keyword", "semantic", or "hybrid"
    - top_k: number of results to return (default: 5)
    """
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query_text = data['query']
        search_type = data.get('search_type', 'hybrid')
        top_k = int(data.get('top_k', 5))
        
        # Create query processor
        processor = li.QueryProcessor(
            index=index,
            embedding_function=generate_simple_embedding
        )
        
        # Execute query
        result = processor.query(
            query_text=query_text,
            search_type=search_type,
            top_k=top_k
        )
        
        # Format results
        formatted_results = []
        for match in result.matches:
            formatted_results.append({
                'doc_id': match.doc_id,
                'score': match.score,
                'content': match.content[:200] + "..." if len(match.content) > 200 else match.content,
                'metadata': match.metadata
            })
        
        return jsonify({
            'success': True,
            'query': query_text,
            'search_type': search_type,
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error searching index: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/documents')
def list_documents():
    """List all documents in the index."""
    try:
        documents = []
        for doc_id in index.doc_ids:
            if doc_id in uploaded_files:
                documents.append(uploaded_files[doc_id])
            else:
                documents.append({
                    'doc_id': doc_id,
                    'content': 'Unknown content',
                    'metadata': {}
                })
        
        return jsonify({
            'success': True,
            'document_count': len(documents),
            'documents': documents
        })
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/document/<doc_id>')
def get_document(doc_id):
    """Get a specific document by ID."""
    try:
        if doc_id in uploaded_files:
            return jsonify({
                'success': True,
                'document': uploaded_files[doc_id]
            })
        else:
            return jsonify({'error': 'Document not found'}), 404
        
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def get_stats():
    """Get index statistics."""
    try:
        stats = index.get_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('web_demo_static', path)


@app.route('/clear', methods=['POST'])
def clear_index():
    """Clear the index."""
    try:
        global index, uploaded_files
        
        # Clear the index
        index.clear()
        
        # Save the empty index
        index.persist(index_dir)
        
        # Clear uploaded files
        uploaded_files = {}
        
        return jsonify({
            'success': True,
            'message': 'Index cleared'
        })
        
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        return jsonify({'error': str(e)}), 500


def create_template_files():
    """Create template files for the web demo."""
    os.makedirs('web_demo_templates', exist_ok=True)
    os.makedirs('web_demo_static', exist_ok=True)
    
    # Create index.html
    with open('web_demo_templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LlamaIndex Demo</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { padding-top: 20px; }
        .container { max-width: 900px; }
        .result-item { margin-bottom: 15px; padding: 10px; border: 1px solid #eee; border-radius: 5px; }
        .score { font-weight: bold; color: #007bff; }
        pre { white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">LlamaIndex Demo</h1>
        
        <ul class="nav nav-tabs mb-4" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="search-tab" data-bs-toggle="tab" data-bs-target="#search" type="button" role="tab">Search</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="upload-tab" data-bs-toggle="tab" data-bs-target="#upload" type="button" role="tab">Upload</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="documents-tab" data-bs-toggle="tab" data-bs-target="#documents" type="button" role="tab">Documents</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="stats-tab" data-bs-toggle="tab" data-bs-target="#stats" type="button" role="tab">Stats</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Search Tab -->
            <div class="tab-pane fade show active" id="search" role="tabpanel">
                <h3>Search Documents</h3>
                <form id="searchForm" class="mb-4">
                    <div class="mb-3">
                        <label for="queryText" class="form-label">Search Query</label>
                        <input type="text" class="form-control" id="queryText" placeholder="Enter your search query">
                    </div>
                    <div class="mb-3">
                        <label for="searchType" class="form-label">Search Type</label>
                        <select class="form-select" id="searchType">
                            <option value="hybrid" selected>Hybrid (Keyword + Semantic)</option>
                            <option value="keyword">Keyword</option>
                            <option value="semantic">Semantic</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="topK" class="form-label">Number of Results</label>
                        <input type="number" class="form-control" id="topK" min="1" max="20" value="5">
                    </div>
                    <button type="submit" class="btn btn-primary">Search</button>
                </form>
                
                <div id="searchResults" class="mt-4">
                    <!-- Results will be populated here -->
                </div>
            </div>
            
            <!-- Upload Tab -->
            <div class="tab-pane fade" id="upload" role="tabpanel">
                <h3>Upload Document</h3>
                <form id="uploadForm" class="mb-4">
                    <div class="mb-3">
                        <label for="docTitle" class="form-label">Document Title</label>
                        <input type="text" class="form-control" id="docTitle" placeholder="Enter document title">
                    </div>
                    <div class="mb-3">
                        <label for="docContent" class="form-label">Document Content</label>
                        <textarea class="form-control" id="docContent" rows="8" placeholder="Enter document content"></textarea>
                    </div>
                    <div class="mb-3">
                        <label for="docMetadata" class="form-label">Metadata (JSON)</label>
                        <textarea class="form-control" id="docMetadata" rows="3" placeholder='{"author": "John Doe", "date": "2023-06-15"}'></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Upload Document</button>
                </form>
                
                <div class="alert alert-success" id="uploadSuccess" style="display:none;">
                    Document uploaded successfully!
                </div>
                <div class="alert alert-danger" id="uploadError" style="display:none;">
                    Error uploading document.
                </div>
            </div>
            
            <!-- Documents Tab -->
            <div class="tab-pane fade" id="documents" role="tabpanel">
                <h3>Document List</h3>
                <div class="d-flex justify-content-between mb-3">
                    <button id="refreshDocs" class="btn btn-outline-primary">Refresh List</button>
                    <button id="clearIndex" class="btn btn-outline-danger">Clear All Documents</button>
                </div>
                <div id="documentList" class="mt-3">
                    <!-- Document list will be populated here -->
                </div>
            </div>
            
            <!-- Stats Tab -->
            <div class="tab-pane fade" id="stats" role="tabpanel">
                <h3>Index Statistics</h3>
                <button id="refreshStats" class="btn btn-outline-primary mb-3">Refresh Stats</button>
                <div id="indexStats" class="mt-3">
                    <!-- Stats will be populated here -->
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Search functionality
        document.getElementById('searchForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const query = document.getElementById('queryText').value;
            const searchType = document.getElementById('searchType').value;
            const topK = document.getElementById('topK').value;
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    search_type: searchType,
                    top_k: topK
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const resultsDiv = document.getElementById('searchResults');
                    resultsDiv.innerHTML = '';
                    
                    if (data.results.length === 0) {
                        resultsDiv.innerHTML = '<div class="alert alert-info">No results found</div>';
                        return;
                    }
                    
                    const resultsHeader = document.createElement('h4');
                    resultsHeader.textContent = `Found ${data.results.length} results:`;
                    resultsDiv.appendChild(resultsHeader);
                    
                    data.results.forEach((result, index) => {
                        const resultItem = document.createElement('div');
                        resultItem.className = 'result-item';
                        
                        const title = result.metadata.title || `Document ${index + 1}`;
                        
                        resultItem.innerHTML = `
                            <h5>${title}</h5>
                            <p><span class="score">Score: ${result.score.toFixed(3)}</span></p>
                            <p><strong>Content:</strong></p>
                            <pre>${result.content}</pre>
                            <p><small>Document ID: ${result.doc_id}</small></p>
                        `;
                        
                        resultsDiv.appendChild(resultItem);
                    });
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while searching');
            });
        });
        
        // Upload functionality
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const title = document.getElementById('docTitle').value;
            const content = document.getElementById('docContent').value;
            let metadata = {};
            
            try {
                const metadataText = document.getElementById('docMetadata').value;
                if (metadataText) {
                    metadata = JSON.parse(metadataText);
                }
            } catch (error) {
                alert('Invalid JSON in metadata field');
                return;
            }
            
            if (!content) {
                alert('Please enter document content');
                return;
            }
            
            fetch('/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: title,
                    content: content,
                    metadata: metadata
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('uploadSuccess').style.display = 'block';
                    document.getElementById('uploadError').style.display = 'none';
                    
                    // Clear form
                    document.getElementById('docTitle').value = '';
                    document.getElementById('docContent').value = '';
                    document.getElementById('docMetadata').value = '';
                    
                    // Hide success message after 3 seconds
                    setTimeout(() => {
                        document.getElementById('uploadSuccess').style.display = 'none';
                    }, 3000);
                } else {
                    document.getElementById('uploadError').textContent = 'Error: ' + data.error;
                    document.getElementById('uploadError').style.display = 'block';
                    document.getElementById('uploadSuccess').style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('uploadError').textContent = 'An error occurred while uploading';
                document.getElementById('uploadError').style.display = 'block';
                document.getElementById('uploadSuccess').style.display = 'none';
            });
        });
        
        // Document list functionality
        function loadDocuments() {
            fetch('/documents')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const docListDiv = document.getElementById('documentList');
                    docListDiv.innerHTML = '';
                    
                    if (data.documents.length === 0) {
                        docListDiv.innerHTML = '<div class="alert alert-info">No documents in the index</div>';
                        return;
                    }
                    
                    data.documents.forEach(doc => {
                        const docItem = document.createElement('div');
                        docItem.className = 'result-item';
                        
                        const title = doc.metadata.title || `Document ${doc.doc_id.substring(0, 8)}`;
                        
                        docItem.innerHTML = `
                            <h5>${title}</h5>
                            <p><strong>Content:</strong></p>
                            <pre>${doc.content}</pre>
                            <p><small>Document ID: ${doc.doc_id}</small></p>
                        `;
                        
                        docListDiv.appendChild(docItem);
                    });
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while loading documents');
            });
        }
        
        document.getElementById('refreshDocs').addEventListener('click', loadDocuments);
        
        // Load documents when tab is shown
        document.querySelector('button[data-bs-target="#documents"]').addEventListener('click', loadDocuments);
        
        // Clear index functionality
        document.getElementById('clearIndex').addEventListener('click', function() {
            if (confirm('Are you sure you want to clear all documents from the index?')) {
                fetch('/clear', {
                    method: 'POST',
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Index cleared successfully');
                        loadDocuments();
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while clearing the index');
                });
            }
        });
        
        // Stats functionality
        function loadStats() {
            fetch('/stats')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const statsDiv = document.getElementById('indexStats');
                    statsDiv.innerHTML = '';
                    
                    const statsTable = document.createElement('table');
                    statsTable.className = 'table';
                    
                    statsTable.innerHTML = `
                        <thead>
                            <tr>
                                <th>Statistic</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                        </tbody>
                    `;
                    
                    const tbody = statsTable.querySelector('tbody');
                    
                    for (const [key, value] of Object.entries(data.stats)) {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${key}</td>
                            <td>${value}</td>
                        `;
                        tbody.appendChild(row);
                    }
                    
                    statsDiv.appendChild(statsTable);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while loading stats');
            });
        }
        
        document.getElementById('refreshStats').addEventListener('click', loadStats);
        
        // Load stats when tab is shown
        document.querySelector('button[data-bs-target="#stats"]').addEventListener('click', loadStats);
        
        // Load stats on initial page load
        document.addEventListener('DOMContentLoaded', loadStats);
    </script>
</body>
</html>
''')


def main():
    """Run the web demo."""
    # Initialize the index
    init_index()
    
    # Create template files if they don't exist
    create_template_files()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    main() 