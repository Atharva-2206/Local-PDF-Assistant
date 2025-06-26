import os
import uuid
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

UPLOAD_DIR = "uploads"
DB_DIR = "vectorstores"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

def save_uploaded_pdf(file):
    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    return filepath, file_id

def extract_text_from_pdf(filepath):
    pdf = fitz.open(filepath)
    return "\n".join(page.get_text() for page in pdf)

def create_vector_store(text, file_id):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    save_path = os.path.join(DB_DIR, file_id)
    vector_store.save_local(save_path)
    return file_id

def process_pdf_file(file):
    filepath, file_id = save_uploaded_pdf(file)
    pdf_text = extract_text_from_pdf(filepath)
    txn_id = create_vector_store(pdf_text, file_id)
    return txn_id

def get_response_from_chat(query, txn_id):
    db_path = os.path.join(DB_DIR, txn_id)
    if not os.path.exists(db_path):
        raise FileNotFoundError("Vector store not found")

    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    llm = Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.run(query)
    return result
