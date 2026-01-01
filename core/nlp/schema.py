"""
Data schemas for scenario analysis

PHASE 2 ENHANCEMENTS:
- First-class time/horizon extraction
- Richer value semantics (target, delta, range, rate)
- Backward compatible with existing code
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ScenarioItem:
    """
    Represents a single parameter/claim within a scenario
    
    PHASE 2: Extended with time extraction and value semantics
    """
    # Core fields (backward compatible)
    parameter: str
    direction: str  # 'increase', 'decrease', 'target', 'stable', 'double', 'halve'
    value: Optional[float] = None
    unit: str = ""
    value_type: str = "absolute"  # PHASE 2: Now includes 'absolute_target', 'delta', 'range', 'rate', 'percent_point', 'direction_only'
    reference: Dict[str, Any] = field(default_factory=dict)  # {'base': 'current', 'year': None}
    confidence: float = 0.5  # 0-1 score
    source_sentence: str = ""
    horizon: Optional[int] = None  # Target year (backward compatible)
    
    # PHASE 2: Time/horizon extraction fields
    baseline_year: Optional[int] = None  # Starting year for change
    target_year: Optional[int] = None  # Target/end year (same as horizon for compatibility)
    time_expression: str = ""  # Raw time phrase, e.g., "by 2040", "2030-2050"
    time_confidence: float = 0.0  # Confidence in time extraction
    
    # PHASE 2: Value semantics fields
    value_min: Optional[float] = None  # For ranges
    value_max: Optional[float] = None  # For ranges
    base_value: Optional[float] = None  # For "from A to B" expressions
    target_value: Optional[float] = None  # For "from A to B" expressions
    is_range: bool = False  # True if value represents a range
    is_rate: bool = False  # True if value is a rate (per year, CAGR, etc.)
    rate_period: str = ""  # Period for rate (e.g., "per year", "annually")
    
    def __post_init__(self):
        """Ensure target_year and horizon stay in sync for backward compatibility"""
        if self.target_year and not self.horizon:
            self.horizon = self.target_year
        elif self.horizon and not self.target_year:
            self.target_year = self.horizon
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            # Backward compatible fields
            'parameter': self.parameter,
            'direction': self.direction,
            'value': self.value,
            'unit': self.unit,
            'value_type': self.value_type,
            'reference': self.reference,
            'confidence': self.confidence,
            'source_sentence': self.source_sentence,
            'horizon': self.horizon,
            
            # PHASE 2: New fields
            'baseline_year': self.baseline_year,
            'target_year': self.target_year,
            'time_expression': self.time_expression,
            'time_confidence': self.time_confidence,
            'value_min': self.value_min,
            'value_max': self.value_max,
            'base_value': self.base_value,
            'target_value': self.target_value,
            'is_range': self.is_range,
            'is_rate': self.is_rate,
            'rate_period': self.rate_period
        }
    
    def get_display_value(self) -> str:
        """Get human-readable value representation"""
        if self.is_range and self.value_min is not None and self.value_max is not None:
            return f"{self.value_min}-{self.value_max} {self.unit}"
        elif self.base_value is not None and self.target_value is not None:
            return f"from {self.base_value} to {self.target_value} {self.unit}"
        elif self.value is not None:
            return f"{self.value} {self.unit}"
        else:
            return self.direction
    
    def get_time_display(self) -> str:
        """Get human-readable time representation"""
        if self.baseline_year and self.target_year:
            return f"{self.baseline_year}-{self.target_year}"
        elif self.target_year:
            return f"by {self.target_year}"
        elif self.time_expression:
            return self.time_expression
        else:
            return "no time specified"


@dataclass
class Scenario:
    """
    Represents a complete scenario
    
    PHASE 2: Enhanced with scenario-level time metadata
    """
    id: str
    title: str
    text: str
    items: List[ScenarioItem] = field(default_factory=list)
    horizon: Optional[int] = None  # Scenario-level target year
    sectors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # PHASE 2: Scenario-level time fields
    baseline_year: Optional[int] = None
    time_confidence: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'text': self.text,
            'items': [item.to_dict() for item in self.items],
            'horizon': self.horizon,
            'sectors': self.sectors,
            'metadata': self.metadata,
            'baseline_year': self.baseline_year,
            'time_confidence': self.time_confidence
        }
    
    def add_item(self, item: ScenarioItem):
        """Add an item to this scenario"""
        self.items.append(item)
    
    def get_parameters(self) -> List[str]:
        """Get list of all parameter names"""
        return [item.parameter for item in self.items]
    
    def get_item_by_parameter(self, parameter: str) -> Optional[ScenarioItem]:
        """Get item by parameter name"""
        for item in self.items:
            if item.parameter.lower() == parameter.lower():
                return item
        return None
    
    def get_time_range(self) -> tuple:
        """Get overall time range for scenario"""
        baseline = self.baseline_year
        target = self.horizon
        
        # Check items for more specific ranges
        for item in self.items:
            if item.baseline_year and (not baseline or item.baseline_year < baseline):
                baseline = item.baseline_year
            if item.target_year and (not target or item.target_year > target):
                target = item.target_year
        
        return (baseline, target)
