"""
Routing logic.
"""

from typing import Literal
from langchain_core.messages import ToolMessage
from src.graph.state import State

def greeter_router(state: State) -> Literal["end_interaction", "call_tool", "go_to_bouncer", "continue_greeter"]:
    messages = state["messages"]
    if not messages:
        return "continue_greeter"
        
    last_message = messages[-1]

    # Security Check
    failed_attempts = 0
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "verify_answer":
            if "VERIFIED" not in msg.content:
                failed_attempts += 1
    
    if failed_attempts >= 3:
        return "end_interaction"

    # Tool Check
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"

    # Success Check
    if isinstance(last_message, ToolMessage) and last_message.name == "verify_answer":
        if "VERIFIED" in last_message.content:
            return "go_to_bouncer"

    # Default
    return "continue_greeter"

def route_after_bouncer(state: State) -> Literal["go_to_specialist", "end_interaction"]:
    # Placeholder
    return "end_interaction"

def route_after_specialist(state: State) -> Literal["end_interaction"]:
    # Placeholder
    return "end_interaction"
