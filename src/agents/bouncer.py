"""
Bouncer agent node.
"""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from src.graph.state import State
from src.graph.config import LLM_MODEL, LLM_TEMPERATURE
from src.tools.bouncer_tools import check_account_status, handoff_to_specialist

SYSTEM_PROMPT = """You are the Bouncer agent for DEUS Bank.
The user has been verified by the Greeter agent.
Your primary task is to determine the customer's account status (Premium, Regular, or Non-Client) using the `check_account_status` tool with their verified IBAN.
You can find the verified IBAN in the 'verify_answer' tool output in the conversation history.

If the status is 'Non-Client':
- Politely inform them they are not a client of DEUS Bank.
- Recommend they contact their bank's support department directly.

If the status is 'Regular':
- Politely inform them they are a regular client.
- Recommend they call our support department at +1112112112 for any requests.
- Regular clients do NOT have access to the Specialist. Always direct them to the support department.

If the status is 'Premium':
- If the user has NOT mentioned a specific request yet:
  - Politely inform them they are a premium client.
  - Ask how you can help them today.
- If the user has a specific high-value request (e.g. yacht insurance, wealth management, real estate):
  - Use the `handoff_to_specialist` tool to transfer them to the Specialist.
- If the user has a general or simple request (not high-value):
  - Politely inform them they are a premium client.
  - Provide the dedicated support number: +1999888999.

Always be polite and professional.
"""

def bouncer_node(state: State):
    """
    Bouncer node that invokes the LLM with the current state messages.
    """
    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    model_with_tools = model.bind_tools([check_account_status, handoff_to_specialist])
    
    messages = state["messages"]
    # We prepend the system prompt to set the context for the bouncer
    invocation_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = model_with_tools.invoke(invocation_messages)
    return {"messages": [response], "active_agent": "bouncer"}
