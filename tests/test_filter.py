"""
Tests for the Filter class
"""
import unittest
from llamaindex.filter import Filter, FilterOperator

class TestFilter(unittest.TestCase):
    """Test cases for the Filter class."""
    
    def test_eq_filter(self):
        """Test equals filter."""
        filter_obj = Filter.eq("field", "value")
        self.assertEqual(filter_obj.operator, FilterOperator.EQ)
        self.assertEqual(filter_obj.field, "field")
        self.assertEqual(filter_obj.value, "value")
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": "value"}))
        self.assertFalse(filter_obj.matches({"field": "other"}))
        self.assertFalse(filter_obj.matches({"other": "value"}))
    
    def test_ne_filter(self):
        """Test not equals filter."""
        filter_obj = Filter.ne("field", "value")
        
        # Test matching
        self.assertFalse(filter_obj.matches({"field": "value"}))
        self.assertTrue(filter_obj.matches({"field": "other"}))
        self.assertFalse(filter_obj.matches({"other": "value"}))
    
    def test_gt_filter(self):
        """Test greater than filter."""
        filter_obj = Filter.gt("field", 10)
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": 20}))
        self.assertFalse(filter_obj.matches({"field": 10}))
        self.assertFalse(filter_obj.matches({"field": 5}))
        self.assertFalse(filter_obj.matches({"other": 20}))
    
    def test_gte_filter(self):
        """Test greater than or equal filter."""
        filter_obj = Filter.gte("field", 10)
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": 20}))
        self.assertTrue(filter_obj.matches({"field": 10}))
        self.assertFalse(filter_obj.matches({"field": 5}))
    
    def test_lt_filter(self):
        """Test less than filter."""
        filter_obj = Filter.lt("field", 10)
        
        # Test matching
        self.assertFalse(filter_obj.matches({"field": 20}))
        self.assertFalse(filter_obj.matches({"field": 10}))
        self.assertTrue(filter_obj.matches({"field": 5}))
    
    def test_lte_filter(self):
        """Test less than or equal filter."""
        filter_obj = Filter.lte("field", 10)
        
        # Test matching
        self.assertFalse(filter_obj.matches({"field": 20}))
        self.assertTrue(filter_obj.matches({"field": 10}))
        self.assertTrue(filter_obj.matches({"field": 5}))
    
    def test_in_filter(self):
        """Test in filter."""
        filter_obj = Filter.in_("field", ["value1", "value2"])
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": "value1"}))
        self.assertTrue(filter_obj.matches({"field": "value2"}))
        self.assertFalse(filter_obj.matches({"field": "value3"}))
    
    def test_nin_filter(self):
        """Test not in filter."""
        filter_obj = Filter.nin("field", ["value1", "value2"])
        
        # Test matching
        self.assertFalse(filter_obj.matches({"field": "value1"}))
        self.assertFalse(filter_obj.matches({"field": "value2"}))
        self.assertTrue(filter_obj.matches({"field": "value3"}))
    
    def test_contains_filter(self):
        """Test contains filter."""
        filter_obj = Filter.contains("field", "substring")
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": "has substring inside"}))
        self.assertFalse(filter_obj.matches({"field": "no match"}))
    
    def test_starts_with_filter(self):
        """Test starts with filter."""
        filter_obj = Filter.starts_with("field", "prefix")
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": "prefix_text"}))
        self.assertFalse(filter_obj.matches({"field": "text_prefix"}))
    
    def test_ends_with_filter(self):
        """Test ends with filter."""
        filter_obj = Filter.ends_with("field", "suffix")
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": "text_suffix"}))
        self.assertFalse(filter_obj.matches({"field": "suffix_text"}))
    
    def test_regex_filter(self):
        """Test regex filter."""
        filter_obj = Filter.regex("field", r"^test_\d+$")
        
        # Test matching
        self.assertTrue(filter_obj.matches({"field": "test_123"}))
        self.assertFalse(filter_obj.matches({"field": "123_test"}))
        self.assertFalse(filter_obj.matches({"field": "test_abc"}))
    
    def test_and_filter(self):
        """Test AND filter."""
        filter_obj = Filter.and_(
            Filter.eq("field1", "value1"),
            Filter.eq("field2", "value2")
        )
        
        # Test matching
        self.assertTrue(filter_obj.matches({
            "field1": "value1",
            "field2": "value2"
        }))
        self.assertFalse(filter_obj.matches({
            "field1": "value1",
            "field2": "other"
        }))
        self.assertFalse(filter_obj.matches({
            "field1": "other",
            "field2": "value2"
        }))
    
    def test_or_filter(self):
        """Test OR filter."""
        filter_obj = Filter.or_(
            Filter.eq("field1", "value1"),
            Filter.eq("field2", "value2")
        )
        
        # Test matching
        self.assertTrue(filter_obj.matches({
            "field1": "value1",
            "field2": "other"
        }))
        self.assertTrue(filter_obj.matches({
            "field1": "other",
            "field2": "value2"
        }))
        self.assertFalse(filter_obj.matches({
            "field1": "other",
            "field2": "other"
        }))
    
    def test_not_filter(self):
        """Test NOT filter."""
        filter_obj = Filter.not_(Filter.eq("field", "value"))
        
        # Test matching
        self.assertFalse(filter_obj.matches({"field": "value"}))
        self.assertTrue(filter_obj.matches({"field": "other"}))
    
    def test_complex_filter(self):
        """Test complex filter combining multiple operations."""
        filter_obj = Filter.and_(
            Filter.or_(
                Filter.eq("category", "technology"),
                Filter.eq("category", "science")
            ),
            Filter.not_(
                Filter.lt("year", 2020)
            )
        )
        
        # Test matching
        self.assertTrue(filter_obj.matches({
            "category": "technology",
            "year": 2023
        }))
        self.assertTrue(filter_obj.matches({
            "category": "science",
            "year": 2020
        }))
        self.assertFalse(filter_obj.matches({
            "category": "technology",
            "year": 2019
        }))
        self.assertFalse(filter_obj.matches({
            "category": "art",
            "year": 2023
        }))
    
    def test_to_dict_from_dict(self):
        """Test Filter serialization to and from dictionary."""
        original_filter = Filter.and_(
            Filter.eq("category", "technology"),
            Filter.gte("year", 2020)
        )
        
        filter_dict = original_filter.to_dict()
        loaded_filter = Filter.from_dict(filter_dict)
        
        # Test that the reconstructed filter behaves the same
        test_data = {"category": "technology", "year": 2022}
        self.assertEqual(original_filter.matches(test_data), loaded_filter.matches(test_data))
        
        test_data = {"category": "other", "year": 2022}
        self.assertEqual(original_filter.matches(test_data), loaded_filter.matches(test_data))

if __name__ == "__main__":
    unittest.main() 