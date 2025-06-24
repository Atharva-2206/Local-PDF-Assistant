import logging
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
class Settings(BaseSettings):
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    VECTOR_STORES_DIR: Path = BASE_DIR / "data" / "vector_stores"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

settings = Settings()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Application State ---
app = FastAPI(title="PDF Processing API", description="Processes PDFs into vector stores", version="1.0.0")

# --- Helper Functions ---
def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(file_path)
    pdf_text = "".join(page.get_text() for page in doc)
    doc.close()
    return pdf_text

# --- Pydantic Models ---
class PDFProcessResponse(BaseModel):
    transaction_id: str
    filename: str
    message: str

# --- API Endpoint ---
@app.post("/process-pdf/", response_model=PDFProcessResponse)
async def process_pdf(file: UploadFile = File(...)):
    transaction_id = str(uuid.uuid4())
    logger.info(f"[{transaction_id}] Starting processing for file: {file.filename}")
    
    file_path = settings.UPLOADS_DIR / f"{transaction_id}_{file.filename}"
    content = await file.read()
    file_path.write_bytes(content)
    
    pdf_text = _extract_text_from_pdf(file_path)
    
    if not pdf_text.strip():
        raise HTTPException(status_code=400, detail="PDF is empty or contains no extractable text.")
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    chunks = splitter.split_text(pdf_text)
    
    # Create vector store
    vector_store_path = str(settings.VECTOR_STORES_DIR / transaction_id)
    embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
    vector_store = await asyncio.to_thread(FAISS.from_texts, chunks, embedding=embeddings)
    await asyncio.to_thread(vector_store.save_local, vector_store_path)
    
    return {"transaction_id": transaction_id, "filename": file.filename, "message": "PDF processed successfully."}
