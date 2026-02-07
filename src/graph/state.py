from enum import Enum
from typing import List, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class State(MessagesState):
    """
    Extended state that includes MessagesState pattern.
    
    Inherits 'messages' from MessagesState.
    """