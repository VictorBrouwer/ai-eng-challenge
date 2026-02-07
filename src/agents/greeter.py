"""
Greeter agent node.

"""

import sys
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from src.graph.state import State
from src.graph.config import LLM_MODEL, LLM_TEMPERATURE
from src.tools.greeterTools import lookup_customer, verify_answer

# ── Structured-output schemas ────────────────────────────────────

class CustomerInfoExtraction(BaseModel):
    """Extracted customer information from the conversation."""
    name: Optional[str] = Field(None, description="The customer's full name if provided")
    phone: Optional[str] = Field(None, description="The customer's phone number if provided")
    iban: Optional[str] = Field(None, description="The customer's IBAN if provided")


class SecretAnswerExtraction(BaseModel):
    """Extracted secret answer from the conversation."""
    answer: str = Field(..., description="The direct answer to the secret question, without extra text.")

SYSTEM_PROMPT = """You are the Greeter agent for DEUS Bank.
Your goal is to identify the customer.
You must verify the customer by matching at least TWO out of THREE details: Name, Phone, IBAN.
Ask for these details if not provided.
Once you have collected at least two details, use the `lookup_customer` tool providing ALL collected details (e.g. Name AND Phone, or Name AND IBAN) to find the customer and get their security question.
Then ask the security question.
When the user answers, use the `verify_answer` tool to check the answer, providing the answer AND the same customer details you used for lookup.
If the answer is correct, you are done.
Do not ask for the security question until you have successfully verified the customer with at least 2 pieces of information (Name, Phone, IBAN).
"""

def greeter_node(state: State):
    """
    Greeter node that invokes the LLM with the current state messages.
    """
    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    tools = [lookup_customer, verify_answer]
    model_with_tools = model.bind_tools(tools)
    
    messages = state["messages"]
    # Prepend system message if not present or just ensure it's there for context
    # Since we are just invoking, let's prepend it to the list of messages passed to the model
    # but not modify the state (unless we return it, which we do).
    # However, usually we don't want to duplicate system messages in the state history.
    # So we'll construct a list for invocation.
    
    invocation_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = model_with_tools.invoke(invocation_messages)
    return {"messages": [response]}
