"""
Tests for the Document class
"""
import unittest
from llamaindex.document import Document

class TestDocument(unittest.TestCase):
    """Test cases for the Document class."""
    
    def test_document_init(self):
        """Test Document initialization."""
        doc = Document(text="Test document")
        self.assertEqual(doc.text, "Test document")
        self.assertIsNotNone(doc.id)
        self.assertTrue(doc.id.startswith("doc-"))
        self.assertIsInstance(doc.metadata, dict)
        self.assertIn("indexed_at", doc.metadata)
    
    def test_document_with_metadata(self):
        """Test Document with custom metadata."""
        metadata = {"source": "test", "author": "unittest"}
        doc = Document(text="Test with metadata", metadata=metadata)
        self.assertEqual(doc.metadata["source"], "test")
        self.assertEqual(doc.metadata["author"], "unittest")
        self.assertIn("indexed_at", doc.metadata)
    
    def test_document_with_custom_id(self):
        """Test Document with a custom ID."""
        doc = Document(text="Test with custom ID", id="custom-123")
        self.assertEqual(doc.id, "custom-123")
    
    def test_document_to_dict(self):
        """Test converting Document to dictionary."""
        doc = Document(
            text="Test to_dict",
            metadata={"source": "test"},
            id="test-dict-123"
        )
        doc_dict = doc.to_dict()
        self.assertEqual(doc_dict["text"], "Test to_dict")
        self.assertEqual(doc_dict["id"], "test-dict-123")
        self.assertEqual(doc_dict["metadata"]["source"], "test")
    
    def test_document_from_dict(self):
        """Test creating Document from dictionary."""
        doc_dict = {
            "text": "Test from_dict",
            "metadata": {"source": "test_dict"},
            "id": "from-dict-123"
        }
        doc = Document.from_dict(doc_dict)
        self.assertEqual(doc.text, "Test from_dict")
        self.assertEqual(doc.id, "from-dict-123")
        self.assertEqual(doc.metadata["source"], "test_dict")
    
    def test_document_to_json_from_json(self):
        """Test Document serialization to and from JSON."""
        original_doc = Document(
            text="Test JSON serialization",
            metadata={"source": "json_test"},
            id="json-123"
        )
        json_str = original_doc.to_json()
        loaded_doc = Document.from_json(json_str)
        
        self.assertEqual(loaded_doc.text, original_doc.text)
        self.assertEqual(loaded_doc.id, original_doc.id)
        self.assertEqual(loaded_doc.metadata["source"], original_doc.metadata["source"])
    
    def test_document_chunk_text(self):
        """Test Document text chunking."""
        # Create a document with a longer text
        long_text = "This is a test document with multiple sentences. " * 10
        doc = Document(text=long_text)
        
        # Test chunking with default parameters
        chunks = doc.chunk_text()
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks, doc.chunks)
        
        # Test chunking with custom parameters
        chunks = doc.chunk_text(chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        # Check that chunk size is roughly respected
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100 + 20)  # Allow for slight variance due to word boundaries

if __name__ == "__main__":
    unittest.main() 