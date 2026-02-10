from typing import Optional

from langgraph.graph import MessagesState


class State(MessagesState):
    """
    Extended state that includes MessagesState pattern.

    Inherits 'messages' from MessagesState.
    """

    active_agent: str = "greeter"
    summary: Optional[str] = None
    failed_verification_attempts: int = 0
    is_verified: bool = False
    conversation_ended: bool = False

