from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api_chat import router as chat_router
from api_pdf_processor import process_pdf_file
from pydantic import BaseModel

class PDFProcessResponse(BaseModel):
    transaction_id: str
    filename: str
    message: str

app = FastAPI()
app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-pdf/", response_model=PDFProcessResponse)
def process_pdf(file: UploadFile = File(...)):
    try:
        txn_id = process_pdf_file(file)
        return PDFProcessResponse(transaction_id=txn_id, filename=file.filename, message="PDF processed successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
