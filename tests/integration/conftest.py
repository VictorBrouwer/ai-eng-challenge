"""
Integration test fixtures for the DEUS Bank multi-agent system.

These tests make real LLM calls via the OpenAI API.
Ensure OPENAI_API_KEY is set in your environment.

Run integration tests:
    pytest tests/integration/ -v

Skip integration tests:
    pytest -m "not integration"
"""

import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph.builder import build_graph


@pytest.fixture(autouse=True)
def _require_openai_api_key():
    """Skip all integration tests when OPENAI_API_KEY is absent."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping integration test")


@pytest.fixture(scope="session")
def graph():
    """Build the LangGraph workflow once and share across all integration tests."""
    return build_graph()


class Conversation:
    """
    Helper for multi-turn conversation testing against the compiled graph.

    Each instance maintains an isolated conversation thread via a unique
    thread_id backed by the graph's MemorySaver checkpointer.
    """

    def __init__(self, graph):
        self._graph = graph
        self.thread_id = str(uuid.uuid4())
        self._config = {"configurable": {"thread_id": self.thread_id}}
        self.state = None
        self.turn_count = 0

    # ── Interaction ──────────────────────────────────────────────────────

    def send(self, message: str) -> str:
        """
        Send a user message, invoke the graph, and return the last AI response.

        The full state is stored on ``self.state`` for further inspection.
        """
        self.turn_count += 1
        self.state = self._graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=self._config,
        )
        return self.last_response

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def last_response(self) -> str:
        """Content of the most recent non-empty AIMessage."""
        if not self.state:
            return ""
        for msg in reversed(self.state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return ""

    @property
    def messages(self) -> list:
        """Full message history."""
        if not self.state:
            return []
        return self.state.get("messages", [])

    @property
    def tool_messages(self) -> list[ToolMessage]:
        """All ToolMessages in the conversation."""
        return [m for m in self.messages if isinstance(m, ToolMessage)]

    @property
    def active_agent(self) -> str:
        """Currently active agent name."""
        if not self.state:
            return "greeter"
        return self.state.get("active_agent", "greeter")

    # ── Tool introspection ───────────────────────────────────────────────

    def was_tool_called(self, tool_name: str) -> bool:
        """Whether *tool_name* was invoked at any point in the conversation."""
        return any(m.name == tool_name for m in self.tool_messages)

    def get_tool_result(self, tool_name: str) -> str | None:
        """Most recent result from *tool_name*, or ``None``."""
        for m in reversed(self.tool_messages):
            if m.name == tool_name:
                return m.content
        return None

    def count_tool_calls(self, tool_name: str) -> int:
        """Number of times *tool_name* was called."""
        return sum(1 for m in self.tool_messages if m.name == tool_name)

    def get_all_tool_results(self, tool_name: str) -> list[str]:
        """All results from *tool_name* in chronological order."""
        return [m.content for m in self.tool_messages if m.name == tool_name]


@pytest.fixture
def conversation(graph):
    """Provide a fresh, isolated Conversation for each test."""
    return Conversation(graph)
