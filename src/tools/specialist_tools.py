from typing import Literal
from langchain_core.tools import tool

VALID_CATEGORIES = ["yacht_insurance", "wealth_management", "real_estate", "general_premium"]

EXPERT_CONTACT = {
    "yacht_insurance": "Yacht & Marine Insurance department at +9876543",
    "wealth_management": "Wealth Management & Advisory department at +1999888",
    "real_estate": "Real Estate Services department at +888666",
    "general_premium": "Premium General Support department at +99887766",
}

@tool
def route_to_expert(category: str) -> str:
    """
    Routes the premium customer to the appropriate expert department.
    
    Args:
        category: One of "yacht_insurance", "wealth_management", "real_estate", or "general_premium".
    """
    if category not in VALID_CATEGORIES:
        return f"Error: Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}"
    
    return f"Routing customer to {EXPERT_CONTACT[category]}. Please inform the customer."

