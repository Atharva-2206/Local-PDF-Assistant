from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app_pdf_processor import get_response_from_chat

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    transaction_id: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat/", response_model=ChatResponse)
def chat_with_pdf(request: ChatRequest):
    try:
        response_text = get_response_from_chat(request.query, request.transaction_id)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
