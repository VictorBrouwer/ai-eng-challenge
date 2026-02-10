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
    
    # Check for failed verification attempts
    last_message = messages[-1]
    current_failures = state.get("failed_verification_attempts", 0)
    is_verified = state.get("is_verified", False)
    
    if isinstance(last_message, ToolMessage) and last_message.name == "verify_answer":
        if "VERIFIED" not in last_message.content:
            current_failures += 1
        else:
            current_failures = 0
            is_verified = True

    if current_failures >= 3:
        standard_message = AIMessage(
            content="I'm sorry, we were unable to verify your identity after several attempts. "
            "Please visit your nearest branch or contact our customer service to complete the verification process."
        )
        return {
            "messages": [standard_message],
            "active_agent": "greeter",
            "failed_verification_attempts": current_failures,
            "is_verified": is_verified,
            "conversation_ended": True,
        }

    response = model_with_tools.invoke(invocation_messages)
    return {
        "messages": [response], 
        "active_agent": "greeter",
        "failed_verification_attempts": current_failures,
        "is_verified": is_verified
    }
