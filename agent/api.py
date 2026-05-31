from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Generator, Dict, Any
import json
import sys
import os
import uvicorn

from llm import XiaoliAgentsLLM
from ReAct import ReActAgent, REACT_PROMPT_TEMPLATE
from tools import ToolExecutor, bocha_web_search_tool

app = FastAPI(title="XiaoLi Agent Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    mode: str  # "agent" or "session"
    question: str

@app.get("/")
async def root():
    return {"message": "Welcome to XiaoLi Agent Chat API"}

@app.post("/chatstream")
async def chatstream(request: ChatRequest):
    """
    Stream chat responses based on mode:
    - "session" mode: Direct LLM response using think interface
    - "agent" mode: ReAct agent response using run interface
    """
    if request.mode not in ["agent", "session"]:
        raise HTTPException(status_code=400, detail="Mode must be either 'agent' or 'session'")
    
    def event_generator():
        """Generator that yields SSE events for streaming response"""
        try:
            if request.mode == "session":
                # Session mode: Direct LLM response with streaming
                llm = XiaoliAgentsLLM()
                
                # Prepare messages for the LLM
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": request.question}
                ]
                
                # Call the LLM think_stream method for true streaming
                for content_chunk in llm.think_stream(messages=messages):
                    if content_chunk:
                        yield f"data: {json.dumps({'type': 'content', 'content': content_chunk}, ensure_ascii=False)}\n\n"
                    
            elif request.mode == "agent":
                # Agent mode: ReAct agent response with streaming
                llm = XiaoliAgentsLLM()
                
                # Initialize tool executor and register tools
                tool_executor = ToolExecutor()
                search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
                tool_executor.registerTool("Search", search_desc, bocha_web_search_tool)
                
                # Create ReAct agent
                agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
                
                # Run the agent and stream the output
                for item in agent.run_stream(request.question):
                    msg_type = item.get('type')
                    if msg_type == 'content':
                        yield f"data: {json.dumps({'type': 'content', 'content': item.get('content', '')}, ensure_ascii=False)}\n\n"
                    elif msg_type == 'info':
                        yield f"data: {json.dumps({'type': 'info', 'content': item.get('content', '')}, ensure_ascii=False)}\n\n"
                    elif msg_type == 'final':
                        yield f"data: {json.dumps({'type': 'content', 'content': item.get('content', '')}, ensure_ascii=False)}\n\n"
                    elif msg_type == 'error':
                        yield f"data: {json.dumps({'type': 'error', 'error': item.get('error', '')}, ensure_ascii=False)}\n\n"
                    elif msg_type == 'done':
                        pass  # Don't send done here, let the finally block handle it"
                    
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done', 'done': True})}\n\n"
    
    # Return the streaming response
    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    # Run the server with UTF-8 encoding
    print("Starting server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_config=None
    )