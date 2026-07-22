import os
import json
import shutil
import time
import base64
import pickle
import fitz  # pymupdf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from sse_starlette.sse import EventSourceResponse
import requests as http_requests

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
IMAGES_DIR = "extracted_images"
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "documents"
METADATA_FILE = "documents.json"
BM25_STORE = "bm25_docs.pkl"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

openai_client = OpenAI()

# Load reranker model lazily (BGE reranker)
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _reranker


def describe_image(image_bytes: bytes, filename: str, page_num: int) -> str:
    """Use GPT-4o to describe an image extracted from a PDF."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this image/graph/chart from page {page_num + 1} of '{filename}' in detail. Include all data, labels, axes, trends, and key takeaways.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


# --- BM25 document store ---

def load_bm25_docs() -> list[Document]:
    if os.path.exists(BM25_STORE):
        with open(BM25_STORE, "rb") as f:
            return pickle.load(f)
    return []


def save_bm25_docs(docs: list[Document]):
    with open(BM25_STORE, "wb") as f:
        pickle.dump(docs, f)


def get_bm25_retriever(k: int = 10) -> BM25Retriever | None:
    docs = load_bm25_docs()
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs, k=k)
    return retriever


# --- Reranking ---

def rerank_documents(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """Rerank documents using BGE reranker."""
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


# --- Web search fallback ---

def web_search(query: str, num_results: int = 3) -> list[Document]:
    """Search the web using DuckDuckGo (no API key needed)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        web_docs = []
        for r in results:
            web_docs.append(
                Document(
                    page_content=f"{r['title']}\n{r['body']}",
                    metadata={"source": r["href"], "type": "web"},
                )
            )
        return web_docs
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


def is_context_sufficient(context: str, question: str) -> bool:
    """Quick check if retrieved context can answer the question."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a relevance checker. Reply ONLY 'yes' or 'no'.",
            },
            {
                "role": "user",
                "content": f"Can the following context sufficiently answer this question?\n\nQuestion: {question}\n\nContext: {context[:2000]}\n\nAnswer yes or no:",
            },
        ],
        max_tokens=3,
    )
    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer


# --- Metadata helpers ---

def load_doc_metadata():
    docs = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            docs = json.load(f)

    tracked_names = {d["filename"] for d in docs}
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            if fname not in tracked_names and (fname.endswith(".pdf") or fname.endswith(".txt")):
                docs.append({
                    "filename": fname,
                    "chunks": "?",
                    "uploaded_at": "previously uploaded",
                })
        if len(docs) > len(tracked_names):
            save_doc_metadata(docs)

    return docs


def save_doc_metadata(docs):
    with open(METADATA_FILE, "w") as f:
        json.dump(docs, f)


# --- Vector store ---

vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        client = QdrantClient(path=QDRANT_PATH)
        embeddings = OpenAIEmbeddings()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings,
            )
    return vectorstore


# --- Upload endpoint ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """SSE endpoint that streams upload/processing progress."""
    global vectorstore

    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def event_stream():
        global vectorstore

        yield json.dumps({"step": "saving", "status": "done", "detail": f"Saved {filename}"})

        # Extract text
        yield json.dumps({"step": "extracting", "status": "running", "detail": "Extracting text from document..."})
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            yield json.dumps({"step": "error", "status": "done", "detail": "Only .pdf and .txt supported"})
            return

        documents = loader.load()
        yield json.dumps({"step": "extracting", "status": "done", "detail": f"Extracted {len(documents)} pages"})

        # Chunk
        yield json.dumps({"step": "chunking", "status": "running", "detail": "Splitting into chunks..."})
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        yield json.dumps({"step": "chunking", "status": "done", "detail": f"Created {len(chunks)} text chunks"})

        # Images
        image_chunks = []
        if filename.endswith(".pdf"):
            yield json.dumps({"step": "images", "status": "running", "detail": "Extracting images from PDF..."})
            doc = fitz.open(file_path)
            total_images = 0
            for page_num in range(len(doc)):
                page = doc[page_num]
                images = page.get_images(full=True)
                for img_idx, img_info in enumerate(images):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    if len(image_bytes) < 5000:
                        continue
                    total_images += 1
                    img_filename = f"{filename}_p{page_num + 1}_img{img_idx + 1}.png"
                    img_path = os.path.join(IMAGES_DIR, img_filename)
                    with open(img_path, "wb") as imgf:
                        imgf.write(image_bytes)

                    yield json.dumps({"step": "images", "status": "running", "detail": f"Describing image {total_images} (page {page_num + 1})..."})
                    try:
                        description = describe_image(image_bytes, filename, page_num)
                        image_chunks.append(
                            Document(
                                page_content=f"[IMAGE from page {page_num + 1}]: {description}",
                                metadata={"source": filename, "page": page_num + 1, "type": "image", "image_file": img_filename},
                            )
                        )
                    except Exception as e:
                        print(f"Failed to describe image {img_filename}: {e}")
            doc.close()
            chunks.extend(image_chunks)
            yield json.dumps({"step": "images", "status": "done", "detail": f"Processed {total_images} images"})

        # Embed into Qdrant
        yield json.dumps({"step": "embedding", "status": "running", "detail": f"Embedding {len(chunks)} chunks..."})
        embeddings = OpenAIEmbeddings()
        vectorstore = QdrantVectorStore.from_documents(
            chunks, embeddings, path=QDRANT_PATH, collection_name=COLLECTION_NAME,
        )
        yield json.dumps({"step": "embedding", "status": "done", "detail": f"Stored {len(chunks)} chunks in Qdrant"})

        # Update BM25 store
        yield json.dumps({"step": "bm25", "status": "running", "detail": "Updating BM25 index..."})
        bm25_docs = load_bm25_docs()
        # Remove old docs from same file
        bm25_docs = [d for d in bm25_docs if d.metadata.get("source") != filename]
        bm25_docs.extend(chunks)
        save_bm25_docs(bm25_docs)
        yield json.dumps({"step": "bm25", "status": "done", "detail": f"BM25 index updated ({len(bm25_docs)} total docs)"})

        # Metadata
        docs_meta = load_doc_metadata()
        docs_meta = [d for d in docs_meta if d["filename"] != filename]
        docs_meta.append({
            "filename": filename,
            "chunks": len(chunks),
            "images": len(image_chunks),
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_doc_metadata(docs_meta)

        yield json.dumps({"step": "complete", "status": "done", "detail": f"Done! {len(chunks)} chunks indexed ({len(image_chunks)} from images)"})

    return EventSourceResponse(event_stream())


# --- Documents endpoints ---

@app.get("/documents")
async def list_documents():
    return {"documents": load_doc_metadata()}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    global vectorstore

    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    if os.path.exists(IMAGES_DIR):
        for img_file in os.listdir(IMAGES_DIR):
            if img_file.startswith(filename):
                os.remove(os.path.join(IMAGES_DIR, img_file))

    # Remove from BM25
    bm25_docs = load_bm25_docs()
    bm25_docs = [d for d in bm25_docs if d.metadata.get("source") != filename]
    save_bm25_docs(bm25_docs)

    docs_meta = load_doc_metadata()
    docs_meta = [d for d in docs_meta if d["filename"] != filename]
    save_doc_metadata(docs_meta)

    vectorstore = None
    return {"message": f"Deleted {filename}"}


# --- Ask endpoint (Hybrid RAG) ---

class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(body: Question):
    """Hybrid RAG: BM25 + Vector search → BGE Rerank → Web fallback → GPT-4o"""
    vs = get_vectorstore()
    if vs is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    async def event_stream():
        question = body.question

        # Step 1: Vector search
        yield json.dumps({"step": "vector_search", "status": "running", "detail": "Semantic search in Qdrant..."})
        start = time.time()
        vector_docs = vs.as_retriever(search_kwargs={"k": 10}).invoke(question)
        vector_time = round(time.time() - start, 2)
        yield json.dumps({"step": "vector_search", "status": "done", "detail": f"Found {len(vector_docs)} docs via vector search ({vector_time}s)"})

        # Step 2: BM25 search
        yield json.dumps({"step": "bm25_search", "status": "running", "detail": "Keyword search (BM25)..."})
        start = time.time()
        bm25_retriever = get_bm25_retriever(k=10)
        bm25_docs = bm25_retriever.invoke(question) if bm25_retriever else []
        bm25_time = round(time.time() - start, 2)
        yield json.dumps({"step": "bm25_search", "status": "done", "detail": f"Found {len(bm25_docs)} docs via BM25 ({bm25_time}s)"})

        # Step 3: Merge and deduplicate
        yield json.dumps({"step": "merge", "status": "running", "detail": "Merging and deduplicating results..."})
        seen_contents = set()
        merged_docs = []
        for doc in vector_docs + bm25_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged_docs.append(doc)
        yield json.dumps({"step": "merge", "status": "done", "detail": f"Merged to {len(merged_docs)} unique chunks"})

        # Step 4: Rerank with BGE
        yield json.dumps({"step": "reranking", "status": "running", "detail": "Reranking with BGE reranker..."})
        start = time.time()
        reranked_docs = rerank_documents(question, merged_docs, top_k=5)
        rerank_time = round(time.time() - start, 2)
        yield json.dumps({"step": "reranking", "status": "done", "detail": f"Reranked to top 5 in {rerank_time}s"})

        # Step 5: Check context sufficiency
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        yield json.dumps({"step": "sufficiency", "status": "running", "detail": "Checking if context is sufficient..."})

        web_docs = []
        sufficient = is_context_sufficient(context, question)

        if sufficient:
            yield json.dumps({"step": "sufficiency", "status": "done", "detail": "Context is sufficient ✓"})
        else:
            yield json.dumps({"step": "sufficiency", "status": "done", "detail": "Context insufficient — searching the web..."})

            # Step 5b: Web search fallback
            yield json.dumps({"step": "web_search", "status": "running", "detail": "Searching the internet..."})
            start = time.time()
            web_docs = web_search(question)
            web_time = round(time.time() - start, 2)
            yield json.dumps({"step": "web_search", "status": "done", "detail": f"Found {len(web_docs)} web results ({web_time}s)"})

            # Add web results to context
            if web_docs:
                web_context = "\n\n".join([doc.page_content for doc in web_docs])
                context = context + "\n\n[WEB SEARCH RESULTS]:\n" + web_context

        # Show sources
        all_sources = reranked_docs + web_docs
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "type": doc.metadata.get("type", "text"),
            }
            for doc in all_sources
        ]
        yield json.dumps({"step": "sources", "status": "done", "detail": f"Using {len(all_sources)} sources", "sources": sources})

        # Step 6: Generate answer with GPT-4o
        yield json.dumps({"step": "generation", "status": "running", "detail": "Generating answer with GPT-4o..."})
        start = time.time()
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        prompt = f"""Use the following context to answer the question. The context may include text, image descriptions, and web search results. If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

        result = llm.invoke(prompt)
        gen_time = round(time.time() - start, 2)

        yield json.dumps({
            "step": "generation",
            "status": "done",
            "detail": f"Generated answer in {gen_time}s",
            "answer": result.content,
        })

        yield json.dumps({"step": "complete", "status": "done", "detail": "Pipeline complete"})

    return EventSourceResponse(event_stream())


@app.get("/health")
async def health():
    return {"status": "ok"}
