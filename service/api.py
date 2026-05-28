"""
HTTP API Service for Qwen Agent
Provides RESTful endpoints to interact with the Qwen language model
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json

# Set HF_ENDPOINT to avoid connection issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize FastAPI app
app = FastAPI(
    title="Xiaoli Agent API",
    description="HTTP interface for Qwen language model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware to ensure UTF-8 encoding
@app.middleware("http")
async def add_utf8_encoding(request, call_next):
    """Middleware to ensure all responses use UTF-8 encoding"""
    response = await call_next(request)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


# Add request body encoding validation
@app.middleware("http")
async def validate_request_encoding(request: Request, call_next):
    """Middleware to validate and ensure request body uses UTF-8 encoding"""
    # Check content type for POST/PUT requests
    if request.method in ["POST", "PUT"]:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            # Ensure charset is utf-8
            if "charset" not in content_type.lower():
                # Log warning but continue processing
                print(f"Warning: Request missing charset in Content-Type header: {content_type}")
    
    response = await call_next(request)
    return response

# Model configuration
MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for model and tokenizer
tokenizer = None
model = None


class ChatMessage(BaseModel):
    """Request model for chat endpoint"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat completion"""
    messages: list[ChatMessage]
    max_new_tokens: int = 512
    temperature: float = 0.7


class ChatResponse(BaseModel):
    """Response model for chat completion"""
    response: str
    model: str
    device: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    device: str
    cuda_available: bool
    model_loaded: bool


def load_model():
    """Load the Qwen model and tokenizer"""
    global tokenizer, model
    
    try:
        print(f"Loading model from {MODEL_ID}...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Load model and move to device
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        device=device,
        cuda_available=torch.cuda.is_available(),
        model_loaded=model is not None and tokenizer is not None
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint
    
    Accepts a list of messages and returns the model's response
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert messages to the format expected by the model
        # Ensure proper UTF-8 encoding for input content
        messages_dict = []
        for msg in request.messages:
            message_data = {
                "role": msg.role,
                "content": msg.content
            }
            # Ensure content is properly encoded as UTF-8 string
            if isinstance(message_data["content"], bytes):
                message_data["content"] = message_data["content"].decode('utf-8')
            messages_dict.append(message_data)
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Encode input
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Generate response
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        
        # Decode only the generated part
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode response with proper encoding
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Ensure UTF-8 encoding
        if isinstance(response_text, bytes):
            response_text = response_text.decode('utf-8')
            
        print(response_text)
        
        return ChatResponse(
            response=response_text,
            model=MODEL_ID,
            device=device
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Xiaoli Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        }
    }


if __name__ == "__main__":
    # Run the server with UTF-8 encoding
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None
    )
