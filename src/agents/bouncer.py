"""
Bouncer agent node.
"""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from src.graph.state import State
from src.graph.config import LLM_MODEL, LLM_TEMPERATURE

SYSTEM_PROMPT = """You are the Bouncer agent for DEUS Bank.
The user has been verified by the Greeter agent.
Your goal is to welcome the verified customer and ask how you can assist them today.
Be polite and professional.
"""

def bouncer_node(state: State):
    """
    Bouncer node that invokes the LLM with the current state messages.
    """
    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    messages = state["messages"]
    # We prepend the system prompt to set the context for the bouncer
    invocation_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = model.invoke(invocation_messages)
    return {"messages": [response]}
