"""
Specialist agent node.

Acts as an intelligent router for premium clients with high-value requests.
Classifies the request and routes to the appropriate expert department.
"""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from src.graph.state import State
from src.graph.config import LLM_MODEL, LLM_TEMPERATURE
from src.tools.specialist_tools import route_to_expert

EXPERT_DEPARTMENTS = {
    "yacht_insurance": "Yacht & Marine Insurance — call +1999888001",
    "wealth_management": "Wealth Management & Advisory — call +1999888002",
    "real_estate": "Real Estate Services — call +1999888003",
    "general_premium": "Premium General Support — call +1999888004",
}

DEPARTMENT_LIST = "\n".join(f"- \"{key}\": {desc}" for key, desc in EXPERT_DEPARTMENTS.items())

SYSTEM_PROMPT = f"""You are the Specialist agent for DEUS Bank.
You are an intelligent router for premium clients who have been transferred to you because they have a high-value request.

Your ONLY task is to:
1. Determine the topic of the customer's request from the conversation history.
2. Use the `route_to_expert` tool to classify it into one of the expert categories below.
3. After the tool confirms the routing, inform the customer which expert department they are being connected to and provide the contact number.

Expert categories:
{DEPARTMENT_LIST}

Guidelines:
- If the request is about boats, yachts, or marine insurance, route to "yacht_insurance".
- If the request is about investments, portfolio management, or financial planning, route to "wealth_management".
- If the request is about property, mortgages, or real estate transactions, route to "real_estate".
- For any other premium request that doesn't clearly fit the above, route to "general_premium".
- You should be able to classify based on the conversation history. If the request is truly unclear, ask ONE clarifying question before routing.
- Always be polite, professional, and concise.
"""

def specialist_node(state: State):
    """
    Specialist node that classifies the request and routes to the right expert.
    """
    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    model_with_tools = model.bind_tools([route_to_expert])
    
    messages = state["messages"]
    invocation_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = model_with_tools.invoke(invocation_messages)
    return {"messages": [response], "active_agent": "specialist"}
