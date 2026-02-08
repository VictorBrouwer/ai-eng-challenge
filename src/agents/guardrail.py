from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.graph.config import LLM_MODEL, LLM_TEMPERATURE
from src.graph.state import State

# 1. Define the Security Policy
SECURITY_POLICY = """
- *The AI must NOT expose or repeat sensitive data (PII, passwords, account numbers).
- *The AI MAY request standard verification details (e.g., phone number or IBAN) when necessary for customer service, but must do so professionally and securely.
- **NO UNAUTHORIZED ACTIONS**: The AI cannot promise loans, approve mortgages, or set interest rates. It must explicitly state it cannot do this if asked.
- **COMPANY TONE**: The response must be polite, professional, and concise.
- **ON-TOPIC**: Avoid political, religious, or controversial discussions.
"""

# 2. Implement the Validation Logic
class SafetyAssessment(BaseModel):
    """Assessment of the safety of a response."""
    is_safe: bool = Field(description="True if the message follows all rules.")
    violation_reason: Optional[str] = Field(description="If unsafe, explain exactly which rule was broken.")
    sanitized_content: Optional[str] = Field(description="If the message was unsafe, provide a polite, corrected version that refuses the request (e.g., 'I cannot approve loans, but I can connect you to a specialist.').")

def validate_response(response_text: str) -> SafetyAssessment:
    """
    Validates a response against the security policy using an LLM.
    
    Args:
        response_text: The text to validate.
        
    Returns:
        SafetyAssessment: The assessment result.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    structured_llm = llm.with_structured_output(SafetyAssessment)
    
    system_prompt = f"""You are a Guardrail Agent for a banking bot.
Your goal is to act as a final firewall before sending any message to the customer.

Start by reviewing the following security policy:
{SECURITY_POLICY}

Analyze the user's proposed response and determine if it adheres to the policy.
If it violates any rule, set is_safe to False, provide the reason, and a sanitized version.
If it is safe, set is_safe to True.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the response to validate:\n{response_text}")
    ])
    
    chain = prompt | structured_llm
    
    return chain.invoke({"response_text": response_text})

def guardrail_node(state: State):
    """
    Guardrail node that validates the last message in the state.
    """
    messages = state["messages"]
    if not messages:
        return {}

    last_message = messages[-1]
    
    # Only validate AIMessages
    if not isinstance(last_message, AIMessage):
        return {}
        
    assessment = validate_response(last_message.content)
    
    if assessment.is_safe:
        return {}
        
    # Replace the unsafe message with the sanitized version
    sanitized_content = assessment.sanitized_content or "I'm sorry, I cannot process that request due to safety policies."
    
    # If the original message has an ID, use it to update.
    # If not, we append the correction.
    msg_id = last_message.id
    
    new_message = AIMessage(content=sanitized_content, id=msg_id)
    
    return {"messages": [new_message]}
