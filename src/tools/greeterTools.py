import json
import os
from typing import Optional
from langchain_core.tools import tool
from src.graph.config import CUSTOMER_DATA_PATH

from src.utils.data import load_customers_data

def _find_customer(customers: list, name: Optional[str] = None, phone: Optional[str] = None, iban: Optional[str] = None):
    """Helper to find a customer matching provided details. Requires at least one detail to match."""
    # Note: validation of "at least two details provided" is done in the tool, 
    # but here we ensure we find a unique customer matching ALL provided non-None details.
    
    for customer in customers:
        is_match = True
        
        if name:
            if customer.get("name", "").lower() != name.strip().lower():
                is_match = False
        
        if phone and is_match:
            if customer.get("phone", "") != phone.strip():
                is_match = False

        if iban and is_match:
            if customer.get("iban", "") != iban.strip():
                is_match = False
        
        if is_match:
            return customer
            
    return None

@tool
def lookup_customer(name: Optional[str] = None, phone: Optional[str] = None, iban: Optional[str] = None) -> str:
    """
    Verifies the customer identity using at least two details (name, phone, IBAN).
    You must provide at least two arguments.
    If verified, returns the security question.
    """
    provided_details = [d for d in [name, phone, iban] if d]
    if len(provided_details) < 2:
        return "Error: You must provide at least two details (Name, Phone, IBAN) to verify the customer."

    try:
        data = load_customers_data()
        customers = data.get("customers", [])
        customer = _find_customer(customers, name=name, phone=phone, iban=iban)
        
        if customer:
            return f"Customer found. Ask this secret question: {customer.get('secret')}"
        return "Customer not found. Please verify the provided details."
    except Exception as e:
        return f"Error looking up customer: {str(e)}"

@tool
def verify_answer(answer: str, name: Optional[str] = None, phone: Optional[str] = None, iban: Optional[str] = None) -> str:
    """
    Checks the answer to the security question for the customer identified by the provided details.
    Provide the same name, phone, or IBAN used in lookup_customer.
    """
    try:
        data = load_customers_data()
        customers = data.get("customers", [])

        # We need at least one identifier to find the customer again
        if not any([name, phone, iban]):
             return "Error: Please provide customer details (Name, Phone, or IBAN) to verify the answer."

        customer = _find_customer(customers, name=name, phone=phone, iban=iban)
        
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
