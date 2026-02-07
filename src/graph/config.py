"""
Configuration management.

Loads configuration from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Customer Data Path
CUSTOMER_DATA_PATH = os.getenv("CUSTOMER_DATA_PATH", "data/customers.json")
