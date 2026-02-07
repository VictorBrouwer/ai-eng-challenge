"""
Customer data loading and management.

Provides functions to load and access customer data from JSON files.
"""

import json
from pathlib import Path

from src.graph.config import CUSTOMER_DATA_PATH

# Cache for loaded customer data
_customers_data = None


def load_customers_data() -> dict:
    """Load customer data from JSON file."""
    global _customers_data
    if _customers_data is None:
        data_path = Path(CUSTOMER_DATA_PATH)
        if not data_path.is_absolute():
            # Resolve relative to project root
            project_root = Path(__file__).parent.parent
            data_path = project_root / data_path
        
        with open(data_path, "r") as f:
            _customers_data = json.load(f)
    return _customers_data
