"""
Bouncer agent node.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from src.graph.state import State
from src.graph.config import LLM_MODEL, LLM_TEMPERATURE
from src.tools.bouncer_tools import check_account_status, handoff_to_specialist
from src.graph.summarization import build_invocation_messages

SYSTEM_PROMPT = """You are the Bouncer agent for DEUS Bank.
The user has been verified by the Greeter agent.
Your primary task is to determine the customer's account status (Premium, Regular, or Non-Client) using the `check_account_status` tool with their verified IBAN.
You can find the verified IBAN in the 'verify_answer' tool output in the conversation history.

If the status is 'Non-Client':
- Politely inform them they are not a client of DEUS Bank.
- Recommend they contact their bank's support department directly.

If the status is 'Regular':
- Politely inform them they are a regular client.
- Recommend they call our support department at +11223344 for any requests.
- Regular clients do NOT have access to the Specialist. Always direct them to the support department.

If the status is 'Premium':
- If the user has NOT mentioned a specific request yet:
  - Politely inform them they are a premium client.
  - Ask how you can help them today.
- If the user has a request:
  - Use the `handoff_to_specialist` tool ONLY. Do NOT generate any text response alongside it.
  - When handing off, your response must contain ONLY the tool call—no explanatory text like "I will transfer you" or similar.

Always be polite and professional.
"""

def bouncer_node(state: State):
    """
    Bouncer node that invokes the LLM with the current state messages.
    """
    model = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    model_with_tools = model.bind_tools([check_account_status, handoff_to_specialist])
    
    messages = state["messages"]
    invocation_messages = build_invocation_messages(
        SYSTEM_PROMPT,
        messages,
        state.get("summary"),
    )
    
    response = model_with_tools.invoke(invocation_messages)

    # When handing off to specialist, do NOT include any text—only the tool call.
    # Enforce this in code since the user should see either a response OR a transfer, not both.
    if isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
        def tool_name(tc):
            return tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        has_handoff = any(tool_name(tc) == "handoff_to_specialist" for tc in response.tool_calls)
        if has_handoff and response.content:
            response = AIMessage(
                content="",
                tool_calls=response.tool_calls,
                id=response.id,
            )

    return {
        "messages": [response],
        "active_agent": "bouncer",
        "is_verified": True,
        "failed_verification_attempts": 0
    }

