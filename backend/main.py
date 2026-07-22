import os
import json
import shutil
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sse_starlette.sse import EventSourceResponse

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "documents"
METADATA_FILE = "documents.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_doc_metadata():
    docs = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            docs = json.load(f)

    # Also pick up any files in uploads/ that aren't tracked yet
    tracked_names = {d["filename"] for d in docs}
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            if fname not in tracked_names and (fname.endswith(".pdf") or fname.endswith(".txt")):
                docs.append({
                    "filename": fname,
                    "chunks": "?",
                    "uploaded_at": "previously uploaded",
                })
        # Persist so next call is fast
        if len(docs) > len(tracked_names):
            save_doc_metadata(docs)

    return docs


def save_doc_metadata(docs):
    with open(METADATA_FILE, "w") as f:
        json.dump(docs, f)

# Global vector store
vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        client = QdrantClient(path=QDRANT_PATH)
        embeddings = OpenAIEmbeddings()
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings,
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

    # Create or update vector store with Qdrant
    embeddings = OpenAIEmbeddings()
    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )

    # Save doc metadata
    docs_meta = load_doc_metadata()
    docs_meta.append({
        "filename": file.filename,
        "chunks": len(chunks),
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_doc_metadata(docs_meta)

    return {"message": f"Uploaded and indexed {file.filename}", "chunks": len(chunks)}


@app.get("/documents")
async def list_documents():
    """Return list of uploaded documents (persists across restarts)."""
    return {"documents": load_doc_metadata()}


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(body: Question):
    """SSE endpoint that streams pipeline steps to the frontend."""
    vs = get_vectorstore()
    if vs is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    async def event_stream():
        # Step 1: Retrieving relevant documents
        yield json.dumps({"step": "retrieval", "status": "running", "detail": "Searching for relevant chunks..."})

        start = time.time()
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(body.question)
        retrieval_time = round(time.time() - start, 2)

        sources = [
            {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
            for doc in docs
        ]
        yield json.dumps({
            "step": "retrieval",
            "status": "done",
            "detail": f"Found {len(docs)} relevant chunks in {retrieval_time}s",
            "sources": sources,
        })

        # Step 2: Generating answer with LLM
        yield json.dumps({"step": "generation", "status": "running", "detail": "Sending to OpenAI GPT-3.5-turbo..."})

        start = time.time()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use the following context to answer the question. If you don't know the answer, say you don't know.

Context:
{context}

Question: {body.question}

Answer:"""

        result = llm.invoke(prompt)
        generation_time = round(time.time() - start, 2)

        yield json.dumps({
            "step": "generation",
            "status": "done",
            "detail": f"Generated answer in {generation_time}s",
            "answer": result.content,
        })

        # Step 3: Done
        yield json.dumps({"step": "complete", "status": "done", "detail": "Pipeline complete"})

    return EventSourceResponse(event_stream())


@app.get("/health")
async def health():
    return {"status": "ok"}
