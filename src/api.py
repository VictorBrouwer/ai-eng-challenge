"""
FastAPI application for the DEUS Bank Customer Support Agent.
Acts as a BFF (Backend for Frontend) proxying requests to the LangGraph API.
"""

import os
import sys
import uuid
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

app = FastAPI(title="DEUS Bank Support Agent API")

# Configuration
LANGGRAPH_API_URL = os.getenv("LANGGRAPH_API_URL", "http://localhost:8123")
ASSISTANT_ID = "agent"

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
    thread_id = request.thread_id or str(uuid.uuid4())
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # 1. Create thread if it doesn't exist
            # We try to create it blindly; if it exists, it might return 409 or 200 depending on API impl.
            # But simpler is to just POST /threads with thread_id. 
            # Most implementations are idempotent or return the existing one.
            try:
                await client.post(f"{LANGGRAPH_API_URL}/threads", json={"thread_id": thread_id})
            except httpx.HTTPStatusError:
                # If it fails, we assume it might exist or we'll find out in the next step
                pass

            # 2. Run the graph and wait for completion
            run_url = f"{LANGGRAPH_API_URL}/threads/{thread_id}/runs/wait"
            payload = {
                "assistant_id": ASSISTANT_ID,
                "input": {
                    "messages": [{"role": "user", "content": request.message}]
                }
            }
            
            run_response = await client.post(run_url, json=payload)
            run_response.raise_for_status()
            
            # 3. Get the final state to retrieve the response
            # Note: /runs/wait might return the state, but fetching state explicitly is safer
            state_url = f"{LANGGRAPH_API_URL}/threads/{thread_id}/state"
            state_response = await client.get(state_url)
            state_response.raise_for_status()
            
            state_data = state_response.json()
            values = state_data.get("values", {})
            messages = values.get("messages", [])
            
            last_response = ""
            if messages:
                # Find the last AIMessage
                for msg in reversed(messages):
                    # Check for "ai" type or "AIMessage" class name if serialized differently
                    msg_type = msg.get("type")
                    if msg_type == "ai":
                        last_response = msg.get("content", "")
                        break

            conversation_ended = values.get("conversation_ended", False)
            return ChatResponse(
                response=last_response,
                thread_id=thread_id,
                conversation_ended=conversation_ended,
            )

        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error communicating with LangGraph API: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"LangGraph API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
