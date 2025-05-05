from fastapi import FastAPI, Request
from pydantic import BaseModel
import asyncio
from batching import queue_request

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    result = await queue_request(request.prompt)
    return {"response": result}
