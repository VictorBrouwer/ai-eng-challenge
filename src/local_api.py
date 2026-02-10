"""
FastAPI application for the DEUS Bank Customer Support Agent.
Running locally without Docker/LangGraph API server.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.graph.builder import build_graph
from langchain_core.messages import HumanMessage, AIMessage
import uuid

app = FastAPI(title="DEUS Bank Support Agent API (Local)")

# Initialize the graph
graph = build_graph()

class ChatRequest(BaseModel):
    message: str
    thread_id: str = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    conversation_ended: bool = False

@app.get("/")
async def root():
    return FileResponse(project_root / 'static' / 'index.html')

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        messages = [HumanMessage(content=request.message)]
        
        # Invoke the graph
        final_state = graph.invoke({"messages": messages}, config=config)
        
        # Get the last AI message
        state_messages = final_state.get("messages", [])
        last_response = ""
        
        if state_messages:
            # We want the last message that is NOT a HumanMessage (which we just sent)
            # Actually, the graph execution adds messages.
            # We want the last message in the history.
            last_msg = state_messages[-1]
            if isinstance(last_msg, AIMessage):
                last_response = last_msg.content
            elif hasattr(last_msg, "content"):
                last_response = str(last_msg.content)
            
        conversation_ended = final_state.get("conversation_ended", False)
        return ChatResponse(
            response=last_response,
            thread_id=thread_id,
            conversation_ended=conversation_ended,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
