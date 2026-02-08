"""
Graph builder.

Constructs and compiles the LangGraph workflow.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.graph.state import State
from src.agents.greeter import greeter_node
from src.agents.bouncer import bouncer_node
from src.agents.specialist import specialist_node
from src.tools.greeter_tools import lookup_customer, verify_answer
from src.tools.bouncer_tools import check_account_status, handoff_to_specialist
from src.tools.specialist_tools import route_to_expert
from src.graph.routing import (
    dispatcher,
    greeter_router,
    route_after_bouncer,
    route_after_bouncer_tools,
    route_after_specialist,
)

def build_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("greeter", greeter_node)
    builder.add_node("bouncer", bouncer_node)
    builder.add_node("specialist", specialist_node)
    
    # Tools
    greeter_tools = [lookup_customer, verify_answer]
    greeter_tool_node = ToolNode(greeter_tools)
    builder.add_node("greeter_tools", greeter_tool_node)
    
    bouncer_tools = [check_account_status, handoff_to_specialist]
    bouncer_tool_node = ToolNode(bouncer_tools)
    builder.add_node("bouncer_tools", bouncer_tool_node)
    
    specialist_tools = [route_to_expert]
    specialist_tool_node = ToolNode(specialist_tools)
    builder.add_node("specialist_tools", specialist_tool_node)

    # Entry point: dispatch to the active agent
    builder.add_conditional_edges(
        START,
        dispatcher,
        {
            "greeter": "greeter",
            "bouncer": "bouncer",
            "specialist": "specialist",
        }
    )
    
    # Greeter conditional edges
    builder.add_conditional_edges(
        "greeter",
        greeter_router,
        {
            "end_interaction": END,
            "call_tool": "greeter_tools",
            "go_to_bouncer": "bouncer",
            "continue_greeter": END,  # Stop and wait for user input
        }
    )

    # After greeter tools execute, always return to greeter to process the result
    builder.add_edge("greeter_tools", "greeter")
    
    # Bouncer conditional edges
    builder.add_conditional_edges(
        "bouncer",
        route_after_bouncer,
        {
            "call_tool": "bouncer_tools",
            "continue_bouncer": END,  # Stop and wait for user input
        }
    )
    
    # After bouncer tools execute, check if it was a handoff or a regular tool
    builder.add_conditional_edges(
        "bouncer_tools",
        route_after_bouncer_tools,
        {
            "go_to_specialist": "specialist",
            "return_to_bouncer": "bouncer",
        }
    )

    # Specialist conditional edges
    builder.add_conditional_edges(
        "specialist",
        route_after_specialist,
        {
            "call_tool": "specialist_tools",
            "continue_specialist": END,  # Stop and wait for user input
        }
    )
    
    # After specialist tools execute, return to specialist to deliver the result
    builder.add_edge("specialist_tools", "specialist")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
