"""
Graph builder.

Constructs and compiles the LangGraph workflow.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.graph.state import State
from src.agents.greeter import greeter_node
from src.tools.greeterTools import lookup_customer, verify_answer
from src.graph.routing import (
    greeter_router,
    route_after_bouncer,
    route_after_specialist,
)

def build_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("greeter", greeter_node)
    
    tools = [lookup_customer, verify_answer]
    tool_node = ToolNode(tools)
    builder.add_node("greeter_tools", tool_node)
    
    # Placeholder nodes
    builder.add_node("bouncer", lambda state: {"messages": []})

    # Edges
    builder.set_entry_point("greeter")
    
    # Greeter conditional edges
    builder.add_conditional_edges(
        "greeter",
        greeter_router,
        {
            "end_interaction": END,
            "call_tool": "greeter_tools",
            "go_to_bouncer": "bouncer",
            "continue_greeter": END  # Stop and wait for user input
        }
    )

    # Tools conditional edges
    builder.add_conditional_edges(
        "greeter_tools",
        greeter_router,
        {
            "end_interaction": END,
            "call_tool": "greeter_tools", # Should usually not happen but safe to have
            "go_to_bouncer": "bouncer",
            "continue_greeter": "greeter" # Loop back to agent to process tool output
        }
    )
    
    # Bouncer edge (placeholder)
    builder.add_edge("bouncer", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
