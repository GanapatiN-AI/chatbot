from fastapi import FastAPI, Request
from pydantic import BaseModel
import asyncio
import os
from chabot_code import qa_agent, session_service, APP_NAME, USER_ID, SESSION_ID
from google.adk.runners import Runner
from google.genai import types

app = FastAPI()

REFERENCE_FILE = "reason_final.txt"

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    # Ensure session is created on startup
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

async def get_reference_text():
    try:
        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    reference_text = await get_reference_text()
    runner = Runner(
        agent=qa_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    user_message = f"Reference text:\n{reference_text}\n\nQuestion: {request.message}"
    content = types.Content(role="user", parts=[types.Part(text=user_message)])
    final_response_text = "Agent did not produce a final response."
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break
    return {"response": final_response_text}
