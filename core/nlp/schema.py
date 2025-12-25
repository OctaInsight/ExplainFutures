"""
Data schemas for scenario analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ScenarioItem:
    """
    Represents a single parameter/claim within a scenario
    """
    parameter: str
    direction: str  # 'increase', 'decrease', 'target', 'stable', 'double', 'halve'
    value: Optional[float] = None
    unit: str = ""
    value_type: str = "absolute"  # 'absolute', 'percent', 'direction_only'
    reference: Dict[str, Any] = field(default_factory=dict)  # {'base': 'current', 'year': None}
    confidence: float = 0.5  # 0-1 score
    source_sentence: str = ""
    horizon: Optional[int] = None  # Target year
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'parameter': self.parameter,
            'direction': self.direction,
            'value': self.value,
            'unit': self.unit,
            'value_type': self.value_type,
            'reference': self.reference,
            'confidence': self.confidence,
            'source_sentence': self.source_sentence,
            'horizon': self.horizon
        }


@dataclass
class Scenario:
    """
    Represents a complete scenario
    """
    id: str
    title: str
    text: str
    items: List[ScenarioItem] = field(default_factory=list)
    horizon: Optional[int] = None
    sectors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'text': self.text,
            'items': [item.to_dict() for item in self.items],
            'horizon': self.horizon,
            'sectors': self.sectors,
            'metadata': self.metadata
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
