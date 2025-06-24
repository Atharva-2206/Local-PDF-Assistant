import logging
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic_settings import BaseSettings
from pathlib import Path
from contextlib import asynccontextmanager

# --- Configuration ---
class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent
    VECTOR_STORES_DIR: Path = BASE_DIR / "data" / "vector_stores"
    LLM_MODEL: str = "llama3"
    EMBEDDING_MODEL: str = "nomic-embed-text"

settings = Settings()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Application State and Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and prompts at application startup."""
    logger.info("Chat API starting up...")
    
    # Updated imports to use OllamaLLM and OllamaEmbeddings
    app.state.llm = OllamaLLM(model=settings.LLM_MODEL)
    app.state.embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
    app.state.prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based ONLY on the following context.
    If the information to answer the question is not in the context, say "I cannot find the answer in the document."
    Do not use any outside knowledge or make up information.

    Context:
    {context}

    Question: {input}
    """)
    
    logger.info(f"Initialized models and prompts. Using LLM: '{settings.LLM_MODEL}'")
    yield
    logger.info("Chat API shutting down.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Chat API for Local PDF Assistant",
    description="Provides a chat interface to query processed PDF documents.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    transaction_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

# --- API Endpoint ---
@app.post("/chat/", response_model=ChatResponse)
async def chat_with_pdf(req: ChatRequest, request: Request):
    transaction_id = req.transaction_id
    question = req.question
    logger.info(f"[{transaction_id}] Received question: '{question}'")

    vector_store_path = settings.VECTOR_STORES_DIR / transaction_id
    if not vector_store_path.exists() or not vector_store_path.is_dir():
        logger.error(f"[{transaction_id}] Vector store not found at path: {vector_store_path}")
        raise HTTPException(
            status_code=404, 
            detail=f"Chat session not found for transaction '{transaction_id}'. Please process the PDF first."
        )

    try:
        # Load the vector store from disk
        vector_store = await asyncio.to_thread(
            FAISS.load_local,
            folder_path=str(vector_store_path),
            embeddings=request.app.state.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create the retrieval chain
        retriever = vector_store.as_retriever()
        document_chain = create_stuff_documents_chain(request.app.state.llm, request.app.state.prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = await retrieval_chain.ainvoke({"input": question})
        
        logger.info(f"[{transaction_id}] Generated answer.")
        return {"answer": response["answer"]}
        
    except Exception as e:
        logger.error(f"[{transaction_id}] Error during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during chat processing: {e}")
