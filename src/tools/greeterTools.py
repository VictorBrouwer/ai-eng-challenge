import json
import os
from langchain_core.tools import tool
from src.graph.config import CUSTOMER_DATA_PATH

from src.utils.data import load_customers_data

def _find_customer(customer_id: str, customers: list):
    """Helper to find a customer by name, phone, or iban."""
    customer_id = customer_id.strip()
    for customer in customers:
        if (customer.get("name") == customer_id or 
            customer.get("phone") == customer_id or 
            customer.get("iban") == customer_id):
            return customer
    return None

@tool
def lookup_customer(customer_id: str) -> str:
    """
    Searches for the customer by name, phone, or IBAN. 
    If found, returns the security question.
    """
    try:
        data = load_customers_data()
        customers = data.get("customers", [])
        customer = _find_customer(customer_id, customers)
        
        if customer:
            return f"Customer found. Ask this security question: {customer.get('secret')}"
        return "Customer not found."
    except Exception as e:
        return f"Error looking up customer: {str(e)}"

@tool
def verify_answer(customer_id: str, answer: str) -> str:
    """
    Checks the answer to the security question for the given customer_id.
    """
    try:
        data = load_customers_data()
        customers = data.get("customers", [])

        customer = _find_customer(customer_id, customers)
        
        if not customer:
            return "Customer not found."
            
        expected_answer = customer.get("answer")
        if expected_answer and answer.strip().lower() == expected_answer.lower():
            # Return JSON string as requested
            return json.dumps({
                "status": "VERIFIED",
                "user_data": customer
            })
        else:
            return "Incorrect answer"
    except Exception as e:
        return f"Error verifying answer: {str(e)}"
