"""
Filter module for LlamaIndex
"""
from typing import Dict, Any, List, Union, Optional, Callable
from enum import Enum
import operator

class FilterOperator(str, Enum):
    """Supported filter operators."""
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NIN = "nin"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    AND = "and"
    OR = "or"
    NOT = "not"

class Filter:
    """
    Filter class for filtering search results.
    
    Filters can be used to narrow down search results based on document metadata.
    """
    
    def __init__(
        self,
        operator: FilterOperator,
        field: Optional[str] = None,
        value: Any = None,
        filters: Optional[List["Filter"]] = None,
    ):
        """
        Initialize a Filter.
        
        Args:
            operator: The filter operator
            field: The metadata field to filter on (not used for logical operators)
            value: The value to compare against (not used for logical operators)
            filters: Nested filters for logical operators (AND, OR, NOT)
        """
        self.operator = operator
        self.field = field
        self.value = value
        self.filters = filters or []
    
    def __repr__(self) -> str:
        """Representation of the filter."""
        if self.operator in [FilterOperator.AND, FilterOperator.OR, FilterOperator.NOT]:
            return f"Filter({self.operator}, filters={self.filters})"
        return f"Filter({self.operator}, field='{self.field}', value={self.value})"
    
    @classmethod
    def eq(cls, field: str, value: Any) -> "Filter":
        """Create an equals filter."""
        return cls(operator=FilterOperator.EQ, field=field, value=value)
    
    @classmethod
    def ne(cls, field: str, value: Any) -> "Filter":
        """Create a not equals filter."""
        return cls(operator=FilterOperator.NE, field=field, value=value)
    
    @classmethod
    def gt(cls, field: str, value: Any) -> "Filter":
        """Create a greater than filter."""
        return cls(operator=FilterOperator.GT, field=field, value=value)
    
    @classmethod
    def gte(cls, field: str, value: Any) -> "Filter":
        """Create a greater than or equal filter."""
        return cls(operator=FilterOperator.GTE, field=field, value=value)
    
    @classmethod
    def lt(cls, field: str, value: Any) -> "Filter":
        """Create a less than filter."""
        return cls(operator=FilterOperator.LT, field=field, value=value)
    
    @classmethod
    def lte(cls, field: str, value: Any) -> "Filter":
        """Create a less than or equal filter."""
        return cls(operator=FilterOperator.LTE, field=field, value=value)
    
    @classmethod
    def in_(cls, field: str, values: List[Any]) -> "Filter":
        """Create an in filter."""
        return cls(operator=FilterOperator.IN, field=field, value=values)
    
    @classmethod
    def nin(cls, field: str, values: List[Any]) -> "Filter":
        """Create a not in filter."""
        return cls(operator=FilterOperator.NIN, field=field, value=values)
    
    @classmethod
    def contains(cls, field: str, value: str) -> "Filter":
        """Create a contains filter."""
        return cls(operator=FilterOperator.CONTAINS, field=field, value=value)
    
    @classmethod
    def starts_with(cls, field: str, value: str) -> "Filter":
        """Create a starts with filter."""
        return cls(operator=FilterOperator.STARTS_WITH, field=field, value=value)
    
    @classmethod
    def ends_with(cls, field: str, value: str) -> "Filter":
        """Create an ends with filter."""
        return cls(operator=FilterOperator.ENDS_WITH, field=field, value=value)
    
    @classmethod
    def regex(cls, field: str, pattern: str) -> "Filter":
        """Create a regex filter."""
        return cls(operator=FilterOperator.REGEX, field=field, value=pattern)
    
    @classmethod
    def and_(cls, *filters: "Filter") -> "Filter":
        """Create an AND filter."""
        return cls(operator=FilterOperator.AND, filters=list(filters))
    
    @classmethod
    def or_(cls, *filters: "Filter") -> "Filter":
        """Create an OR filter."""
        return cls(operator=FilterOperator.OR, filters=list(filters))
    
    @classmethod
    def not_(cls, filter_: "Filter") -> "Filter":
        """Create a NOT filter."""
        return cls(operator=FilterOperator.NOT, filters=[filter_])
    
    def matches(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if the given metadata matches this filter.
        
        Args:
            metadata: The document metadata to check against
            
        Returns:
            Whether the metadata matches the filter
        """
        # Logical operators
        if self.operator == FilterOperator.AND:
            return all(f.matches(metadata) for f in self.filters)
        elif self.operator == FilterOperator.OR:
            return any(f.matches(metadata) for f in self.filters)
        elif self.operator == FilterOperator.NOT:
            return not self.filters[0].matches(metadata)
        
        # Field doesn't exist in metadata
        if self.field not in metadata:
            return False
        
        # Get field value
        field_value = metadata[self.field]
        
        # Comparison operators
        if self.operator == FilterOperator.EQ:
            return field_value == self.value
        elif self.operator == FilterOperator.NE:
            return field_value != self.value
        elif self.operator == FilterOperator.GT:
            return field_value > self.value
        elif self.operator == FilterOperator.GTE:
            return field_value >= self.value
        elif self.operator == FilterOperator.LT:
            return field_value < self.value
        elif self.operator == FilterOperator.LTE:
            return field_value <= self.value
        elif self.operator == FilterOperator.IN:
            return field_value in self.value
        elif self.operator == FilterOperator.NIN:
            return field_value not in self.value
        
        # String operators
        if not isinstance(field_value, str):
            field_value = str(field_value)
            
        if self.operator == FilterOperator.CONTAINS:
            return self.value in field_value
        elif self.operator == FilterOperator.STARTS_WITH:
            return field_value.startswith(self.value)
        elif self.operator == FilterOperator.ENDS_WITH:
            return field_value.endswith(self.value)
        elif self.operator == FilterOperator.REGEX:
            import re
            return bool(re.search(self.value, field_value))
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to a dictionary."""
        result = {"operator": self.operator.value}
        
        if self.operator in [FilterOperator.AND, FilterOperator.OR, FilterOperator.NOT]:
            result["filters"] = [f.to_dict() for f in self.filters]
        else:
            result["field"] = self.field
            result["value"] = self.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Filter":
        """Create a Filter from a dictionary."""
        operator = FilterOperator(data["operator"])
        
        if operator in [FilterOperator.AND, FilterOperator.OR, FilterOperator.NOT]:
            filters = [cls.from_dict(f) for f in data["filters"]]
            return cls(operator=operator, filters=filters)
        else:
            return cls(
                operator=operator,
                field=data["field"],
                value=data["value"]
            ) 