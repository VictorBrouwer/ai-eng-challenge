"""
Graph builder for the DEUS Bank AI agent system.

Constructs the LangGraph workflow that orchestrates the greeter, bouncer,
and specialist agents with their respective tools and routing logic.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import State
from src.graph.routing import (
    await_input_node,
    route_after_await_input,
    greeter_router,
    route_after_greeter_tools,
    route_after_bouncer,
    route_after_bouncer_tools,
    route_after_specialist,
)
from src.agents.greeter import greeter_node
from src.agents.bouncer import bouncer_node
from src.agents.specialist import specialist_node
from src.tools.greeter_tools import lookup_customer, verify_answer
from src.tools.bouncer_tools import check_account_status, handoff_to_specialist
from src.tools.specialist_tools import route_to_expert
from langgraph.prebuilt import ToolNode


def build_graph():
    """
    Build and compile the LangGraph workflow.
    
    Returns:
        Compiled LangGraph application ready for invocation.
    """
    builder = StateGraph(State)

    # ── Nodes ────────────────────────────────────────────────────────────
    builder.add_node("await_input", await_input_node)
    builder.add_node("greeter", greeter_node)
    builder.add_node("bouncer", bouncer_node)
    builder.add_node("specialist", specialist_node)

    # Tool execution nodes
    greeter_tools = ToolNode([lookup_customer, verify_answer])
    bouncer_tools = ToolNode([check_account_status, handoff_to_specialist])
    specialist_tools = ToolNode([route_to_expert])

    builder.add_node("greeter_tools", greeter_tools)
    builder.add_node("bouncer_tools", bouncer_tools)
    builder.add_node("specialist_tools", specialist_tools)

    # ── Entry point ─────────────────────────────────────────────────────
    builder.set_entry_point("await_input")

    # ── Await input routing ─────────────────────────────────────────────
    builder.add_conditional_edges(
        "await_input",
        route_after_await_input,
        {
            "greeter": "greeter",
            "bouncer": "bouncer",
            "specialist": "specialist",
            "__end__": END,
        },
    )

    # ── Greeter edges ────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "greeter",
        greeter_router,
        {
            "call_tool": "greeter_tools",
            "go_to_bouncer": "bouncer",
            "continue_greeter": "await_input",  # consume next user message
            "end_interaction": END,
        },
    )

    # After greeter tools execute, go straight to bouncer on success
    builder.add_conditional_edges(
        "greeter_tools",
        route_after_greeter_tools,
        {
            "go_to_bouncer": "bouncer",
            "return_to_greeter": "greeter",
        },
    )

    # ── Bouncer edges ────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "bouncer",
        route_after_bouncer,
        {
            "call_tool": "bouncer_tools",
            "continue_bouncer": "await_input",  # consume next user message
            "end_interaction": END,
        },
    )

    # After bouncer tools execute, check if we need to hand off to specialist
    builder.add_conditional_edges(
        "bouncer_tools",
        route_after_bouncer_tools,
        {
            "go_to_specialist": "specialist",
            "return_to_bouncer": "bouncer",
        },
    )

    # ── Specialist edges ──────────────────────────────────────────────
    builder.add_conditional_edges(
        "specialist",
        route_after_specialist,
        {
            "call_tool": "specialist_tools",
            "continue_specialist": "await_input",  # consume next user message
            "end_interaction": END,
        },
    )

    # After specialist tools execute, return to specialist to deliver result
    builder.add_edge("specialist_tools", "specialist")

    # ── Compile ───────────────────────────────────────────────────────
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
