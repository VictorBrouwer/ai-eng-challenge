"""
Routing logic.
"""

from typing import Literal
from langchain_core.messages import ToolMessage
from src.graph.state import State


def dispatcher(state: State) -> Literal["greeter", "bouncer", "specialist"]:
    """
    Entry-point router. Reads active_agent from state and dispatches
    to the correct agent node on each graph invocation.
    """
    return state.get("active_agent", "greeter")


def greeter_router(state: State) -> Literal["end_interaction", "call_tool", "go_to_bouncer", "continue_greeter"]:
    messages = state["messages"]
    if not messages:
        return "continue_greeter"
        
    last_message = messages[-1]

    # Failed verification attempts check
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

    # Success Check (anywhere in history)
    is_verified = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "verify_answer":
            if "VERIFIED" in msg.content:
                is_verified = True
                break
    
    if is_verified:
        return "go_to_bouncer"

    # Default: wait for user input
    return "continue_greeter"


def route_after_bouncer(state: State) -> Literal["call_tool", "continue_bouncer"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # All tool calls (including handoff_to_specialist) go through bouncer_tools
    # so the ToolMessage response is always added to the history.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    
    # Wait for user input (or conversation naturally ends)
    return "continue_bouncer"


def route_after_bouncer_tools(state: State) -> Literal["go_to_specialist", "return_to_bouncer"]:
    """
    After bouncer_tools executes, check whether the tool that just ran
    was handoff_to_specialist. If so, route to specialist; otherwise
    return to bouncer to process the tool result.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, ToolMessage) and last_message.name == "handoff_to_specialist":
        return "go_to_specialist"
    
    return "return_to_bouncer"


def route_after_specialist(state: State) -> Literal["call_tool", "continue_specialist"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    
    # Wait for user input or conversation ends naturally
    return "continue_specialist"
