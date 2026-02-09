"""
Greeter agent node.

"""

import sys
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from src.graph.state import State
from src.graph.config import LLM_MODEL, LLM_TEMPERATURE
from src.tools.greeter_tools import lookup_customer, verify_answer
from src.graph.summarization import build_invocation_messages


# SYSTEM_PROMPT = """You are the Greeter agent for DEUS Bank.
# Your goal is to identify the customer.
# You must verify the customer by matching at least TWO out of THREE details: Name, Phone, IBAN.
# Ask for these details if only one or none of the details are provided.
# If you have collected at least two details, use the `lookup_customer` tool providing ALL collected details (e.g. Name AND Phone, or Name AND IBAN) to find the customer and get their secret question.
# Then ask the secret question.
# When the user answers, use the `verify_answer` tool to check the answer, providing the answer AND the same customer details you used for lookup.
# If the answer is correct, you are done.
# Do not ask for the secret question until you have successfully verified the customer with at least 2 pieces of information (Name, Phone, IBAN).
# """

SYSTEM_PROMPT = """You are the Greeter agent for DEUS Bank.

Goal: verify the customer.

Use exactly this logic:

1. Count provided details: Name, Phone, IBAN.
2. If fewer than 2 → ask ONLY for missing details. Do not use tools.
3. If 2 or more → immediately call `lookup_customer` with ALL collected details. Do not ask for anything else.
4. After lookup → ask the secret question.
5. After the answer → call `verify_answer` with the answer and the SAME details.

Rules:
Never ask for more details once you have at least 2.
Never ask the secret question before lookup.
Always use tools when eligible.
Be polite and professional.
"""

def greeter_node(state: State):
    """
    Greeter node that invokes the LLM with the current state messages.
    """
    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    tools = [lookup_customer, verify_answer]
    model_with_tools = model.bind_tools(tools)
    
    messages = state["messages"]
    invocation_messages = build_invocation_messages(
        SYSTEM_PROMPT,
        messages,
        state.get("summary"),
    )
    
    response = model_with_tools.invoke(invocation_messages)
    return {"messages": [response], "active_agent": "greeter"}
