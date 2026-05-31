from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Generator, Dict, Any
import json
import sys
import os
import uvicorn

# 添加agent目录到Python路径，以便可以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

from agent.llm import XiaoliAgentsLLM
from agent.ReAct import ReActAgent, REACT_PROMPT_TEMPLATE
from agent.tools import ToolExecutor, bocha_web_search_tool

app = FastAPI(title="XiaoLi Agent Chat API", version="1.0.0")

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
                # Session mode: Direct LLM response
                llm = XiaoliAgentsLLM()
                
                # Prepare messages for the LLM
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": request.question}
                ]
                
                # Call the LLM think method and stream responses
                response = llm.think(messages=messages)
                if response:
                    yield f"data: {json.dumps({'type': 'content', 'content': response})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to get response from LLM'})}\n\n"
                    
            elif request.mode == "agent":
                # Agent mode: ReAct agent response
                llm = XiaoliAgentsLLM()
                
                # Initialize tool executor and register tools
                tool_executor = ToolExecutor()
                search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
                tool_executor.registerTool("Search", search_desc, bocha_web_search_tool)
                
                # Create ReAct agent
                agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
                
                # Run the agent and capture the output
                result = agent.run(request.question)
                
                if result:
                    yield f"data: {json.dumps({'type': 'content', 'content': result})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Agent failed to produce a result'})}\n\n"
                    
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done', 'done': True})}\n\n"
    
    # Return the streaming response
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    # According to the HTTP service development specification, deploy on host 0.0.0.0 with auto-reload enabled
     uvicorn.run(
        "app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None
    )