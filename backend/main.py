import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global vector store
vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None and os.path.exists(VECTORSTORE_DIR):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
    return vectorstore


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load document based on file type
    if file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.filename.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Create or update vector store
    embeddings = OpenAIEmbeddings()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)

    # Persist
    vectorstore.save_local(VECTORSTORE_DIR)

    return {"message": f"Uploaded and indexed {file.filename}", "chunks": len(chunks)}


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(body: Question):
    vs = get_vectorstore()
    if vs is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 3}),
    )

    result = qa_chain.invoke({"query": body.question})
    return {"answer": result["result"]}


@app.get("/health")
async def health():
    return {"status": "ok"}
