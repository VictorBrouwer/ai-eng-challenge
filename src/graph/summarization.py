"""
Conversation summarization helpers.
"""

from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from src.graph.config import LLM_MODEL, LLM_TEMPERATURE
from src.graph.state import State


def build_invocation_messages(
    system_prompt: str,
    messages: List[BaseMessage],
    summary: Optional[str],
) -> List[BaseMessage]:
    """
    Build messages for an agent call, injecting the optional summary.
    """

    invocation_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    if summary:
        invocation_messages.append(
            SystemMessage(content=f"Summary of conversation earlier: {summary}")
        )

    invocation_messages.extend(messages)
    return invocation_messages


def summarize_conversation(state: State) -> dict:
    """
    Summarize conversation history and prune older messages.
    """

    summary = state.get("summary") or ""
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state.get("messages", [])
    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    response = model.invoke(messages + [HumanMessage(content=summary_message)])

    keep_count = min(2, len(messages))
    keep_indices = set(range(len(messages) - keep_count, len(messages)))

    for index in list(keep_indices):
        message = messages[index]
        if not isinstance(message, ToolMessage):
            continue

        tool_call_id = message.tool_call_id
        if not tool_call_id:
            keep_indices.discard(index)
            continue

        tool_call_index = None
        for search_index in range(index - 1, -1, -1):
            candidate = messages[search_index]
            if isinstance(candidate, AIMessage) and getattr(candidate, "tool_calls", None):
                for call in candidate.tool_calls:
                    if call.get("id") == tool_call_id:
                        tool_call_index = search_index
                        break
            if tool_call_index is not None:
                break

        if tool_call_index is None:
            keep_indices.discard(index)
        else:
            keep_indices.add(tool_call_index)

    delete_messages = [
        RemoveMessage(id=message.id)
        for index, message in enumerate(messages)
        if index not in keep_indices
    ]
    return {"summary": response.content, "messages": delete_messages}
