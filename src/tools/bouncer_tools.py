import json
from typing import Optional
from langchain_core.tools import tool
from src.utils.data import load_customers_data

@tool
def check_account_status(iban: str) -> str:
    """
    Checks the account status (Premium, Regular, or Non-Client) based on the IBAN.
    Returns: "Premium", "Regular", or "Non-Client".
    """
    try:
        data = load_customers_data()
        accounts = data.get("accounts", [])
        
        # clean the iban
        clean_iban = iban.strip()
        
        for account in accounts:
            if account.get("iban") == clean_iban:
                is_premium = account.get("premium", False)
                return "Premium" if is_premium else "Regular"
                
        return "Non-Client"
    except Exception as e:
        return f"Error checking account status: {str(e)}"

@tool
def handoff_to_specialist() -> str:
    """
    Hands off the conversation to the Specialist agent for high-value requests.
    Use this ONLY when a Premium client has a specific high-value request
    (e.g., yacht insurance, wealth management, real estate).
    """
    return "Handing off to Specialist."