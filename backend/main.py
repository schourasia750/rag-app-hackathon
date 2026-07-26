import os
import json
import shutil
import time
import base64
import pickle
import fitz  # pymupdf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
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
import re

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

# Serve extracted images as static files
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

openai_client = OpenAI()

# Lazy-loaded reranker
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    return _reranker


# --- Chat history (in-memory, per session) ---
chat_history = []  # list of {"role": "user"/"assistant", "content": "..."}
MAX_HISTORY = 10


def add_to_history(role: str, content: str):
    chat_history.append({"role": role, "content": content})
    if len(chat_history) > MAX_HISTORY * 2:
        del chat_history[:2]


def get_history_context() -> str:
    if not chat_history:
        return ""
    # Only last 3 exchanges, brief summaries
    recent = chat_history[-6:]
    lines = []
    for msg in recent:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        # Truncate assistant messages to avoid polluting context
        content = msg['content'][:300] if msg["role"] == "assistant" else msg['content']
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


# --- Image extraction on-demand from PDF ---

def extract_page_images(filename: str, page_num: int) -> list[dict]:
    """Extract all images from a specific page of a PDF. Returns list of {url, page}."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return []

    doc = fitz.open(file_path)
    if page_num < 1 or page_num > len(doc):
        doc.close()
        return []

    page = doc[page_num - 1]
    images_found = []
    img_list = page.get_images(full=True)

    for img_idx, img_info in enumerate(img_list):
        xref = img_info[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        if len(image_bytes) < 5000:
            continue

        img_filename = f"{filename}_p{page_num}_img{img_idx + 1}.png"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        images_found.append({
            "url": f"http://localhost:8000/images/{img_filename}",
            "page": page_num,
            "source": filename,
        })

    doc.close()
    return images_found


# --- GPT image description at upload time ---

def describe_image(image_bytes: bytes, filename: str, page_num: int) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe this image/graph/chart from page {page_num + 1} of '{filename}' in detail. Include all data, labels, axes, trends, and key takeaways."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=1000,
    )
    return response.choices[0].message.content


# --- BM25 ---

def load_bm25_docs() -> list[Document]:
    if os.path.exists(BM25_STORE):
        with open(BM25_STORE, "rb") as f:
            return pickle.load(f)
    return []


def save_bm25_docs(docs: list[Document]):
    with open(BM25_STORE, "wb") as f:
        pickle.dump(docs, f)


def get_bm25_retriever(k: int = 10):
    docs = load_bm25_docs()
    if not docs:
        # Try to rebuild from vectorstore if available
        docs = rebuild_bm25_from_uploads()
    if not docs:
        return None
    return BM25Retriever.from_documents(docs, k=k)


def rebuild_bm25_from_uploads() -> list[Document]:
    """Rebuild BM25 index from all uploaded files."""
    all_docs = []
    if not os.path.exists(UPLOAD_DIR):
        return []
    for fname in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, fname)
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif fname.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        try:
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            all_docs.extend(chunks)
        except Exception as e:
            print(f"Failed to load {fname} for BM25: {e}")
    if all_docs:
        save_bm25_docs(all_docs)
    return all_docs


# --- Reranking ---

def rerank_documents(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


# --- Web search ---

def web_search(query: str, num_results: int = 3) -> list[Document]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        return [
            Document(
                page_content=f"{r['title']}\n{r['body']}",
                metadata={"source": r["href"], "type": "web"},
            )
            for r in results
        ]
    except Exception as e:
        print(f"Web search failed: {e}")
        return []


def is_context_sufficient(context: str, question: str) -> bool:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a relevance checker. Reply ONLY 'yes' or 'no'."},
            {"role": "user", "content": f"Can the following context from uploaded documents sufficiently answer this question? Only say 'no' if the context is completely unrelated or empty.\n\nQuestion: {question}\n\nContext: {context[:2000]}\n\nAnswer yes or no:"},
        ],
        max_tokens=3,
    )
    return "yes" in response.choices[0].message.content.strip().lower()


# --- Metadata ---

def load_doc_metadata():
    docs = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            docs = json.load(f)
    tracked_names = {d["filename"] for d in docs}
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            if fname not in tracked_names and (fname.endswith(".pdf") or fname.endswith(".txt")):
                docs.append({"filename": fname, "chunks": "?", "uploaded_at": "previously uploaded"})
        if len(docs) > len(tracked_names):
            save_doc_metadata(docs)
    return docs


def save_doc_metadata(docs):
    with open(METADATA_FILE, "w") as f:
        json.dump(docs, f)


# --- Vector store ---

# Single shared Qdrant client to avoid lock conflicts (lazy init)
qdrant_client = None
vectorstore = None


def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(path=QDRANT_PATH)
    return qdrant_client


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        client = get_qdrant_client()
        embeddings = OpenAIEmbeddings()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    return vectorstore


# --- Upload endpoint ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore
    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def event_stream():
        global vectorstore

        yield json.dumps({"step": "saving", "status": "done", "detail": f"Saved {filename}"})

        yield json.dumps({"step": "extracting", "status": "running", "detail": "Extracting text..."})
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            yield json.dumps({"step": "error", "status": "done", "detail": "Only .pdf and .txt supported"})
            return
        documents = loader.load()
        yield json.dumps({"step": "extracting", "status": "done", "detail": f"Extracted {len(documents)} pages"})

        yield json.dumps({"step": "chunking", "status": "running", "detail": "Splitting into chunks..."})
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        yield json.dumps({"step": "chunking", "status": "done", "detail": f"Created {len(chunks)} text chunks"})

        # Image extraction and description
        image_chunks = []
        if filename.endswith(".pdf"):
            yield json.dumps({"step": "images", "status": "running", "detail": "Extracting images..."})
            doc = fitz.open(file_path)
            total_images = 0
            for page_num in range(len(doc)):
                page = doc[page_num]
                img_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(img_list):
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
                        image_chunks.append(Document(
                            page_content=f"[IMAGE on page {page_num + 1}, image #{img_idx + 1}]: {description}",
                            metadata={"source": filename, "page": page_num + 1, "type": "image", "image_file": img_filename},
                        ))
                    except Exception as e:
                        print(f"Failed: {e}")
            doc.close()
            chunks.extend(image_chunks)
            yield json.dumps({"step": "images", "status": "done", "detail": f"Processed {total_images} images"})

        yield json.dumps({"step": "embedding", "status": "running", "detail": f"Embedding {len(chunks)} chunks..."})
        embeddings = OpenAIEmbeddings()
        client = get_qdrant_client()

        # Check if collection exists, if not create it
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            # Create collection with first batch
            from qdrant_client.models import VectorParams, Distance
            # Get embedding dimension by embedding a test string
            test_embedding = embeddings.embed_query("test")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=len(test_embedding), distance=Distance.COSINE),
            )

        # Use existing vectorstore or create new one
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        # Add documents to the vectorstore
        vectorstore.add_documents(chunks)
        yield json.dumps({"step": "embedding", "status": "done", "detail": f"Stored in Qdrant"})

        yield json.dumps({"step": "bm25", "status": "running", "detail": "Updating BM25 index..."})
        bm25_docs = load_bm25_docs()
        bm25_docs = [d for d in bm25_docs if d.metadata.get("source") != filename]
        bm25_docs.extend(chunks)
        save_bm25_docs(bm25_docs)
        yield json.dumps({"step": "bm25", "status": "done", "detail": "BM25 updated"})

        docs_meta = load_doc_metadata()
        docs_meta = [d for d in docs_meta if d["filename"] != filename]
        docs_meta.append({"filename": filename, "chunks": len(chunks), "images": len(image_chunks), "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        save_doc_metadata(docs_meta)

        yield json.dumps({"step": "complete", "status": "done", "detail": f"Done! {len(chunks)} chunks ({len(image_chunks)} images)"})

    return EventSourceResponse(event_stream())


# --- Documents ---

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
    bm25_docs = load_bm25_docs()
    bm25_docs = [d for d in bm25_docs if d.metadata.get("source") != filename]
    save_bm25_docs(bm25_docs)
    docs_meta = load_doc_metadata()
    docs_meta = [d for d in docs_meta if d["filename"] != filename]
    save_doc_metadata(docs_meta)
    vectorstore = None
    return {"message": f"Deleted {filename}"}


# --- Ask endpoint ---

class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(body: Question):
    vs = get_vectorstore()
    if vs is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    async def event_stream():
        question = body.question
        add_to_history("user", question)

        # Step 1: Vector search
        yield json.dumps({"step": "vector_search", "status": "running", "detail": "Semantic search..."})
        start = time.time()
        vector_docs = vs.as_retriever(search_kwargs={"k": 10}).invoke(question)
        yield json.dumps({"step": "vector_search", "status": "done", "detail": f"Found {len(vector_docs)} docs ({round(time.time()-start,2)}s)"})

        # Step 2: BM25
        yield json.dumps({"step": "bm25_search", "status": "running", "detail": "Keyword search (BM25)..."})
        start = time.time()
        bm25_retriever = get_bm25_retriever(k=10)
        bm25_docs = bm25_retriever.invoke(question) if bm25_retriever else []
        yield json.dumps({"step": "bm25_search", "status": "done", "detail": f"Found {len(bm25_docs)} docs ({round(time.time()-start,2)}s)"})

        # Step 3: Merge
        seen = set()
        merged = []
        for doc in vector_docs + bm25_docs:
            h = hash(doc.page_content[:200])
            if h not in seen:
                seen.add(h)
                merged.append(doc)
        yield json.dumps({"step": "merge", "status": "done", "detail": f"{len(merged)} unique chunks"})

        # Step 4: Rerank
        yield json.dumps({"step": "reranking", "status": "running", "detail": "Reranking with BGE..."})
        start = time.time()
        reranked = rerank_documents(question, merged, top_k=7)
        yield json.dumps({"step": "reranking", "status": "done", "detail": f"Top {len(reranked)} selected ({round(time.time()-start,2)}s)"})

        # Find image chunks from same pages as reranked text chunks
        reranked_pages = set()
        for doc in reranked:
            p = doc.metadata.get("page")
            s = doc.metadata.get("source")
            if p and s:
                reranked_pages.add((s, p))
                # Also add adjacent pages (figures often on next page)
                reranked_pages.add((s, p + 1))
                reranked_pages.add((s, p - 1))

        # Pull matching image chunks from all stored docs
        all_stored = load_bm25_docs()
        related_images = [
            doc for doc in all_stored
            if doc.metadata.get("type") == "image"
            and (doc.metadata.get("source"), doc.metadata.get("page")) in reranked_pages
        ]
        # Also from vector results
        for doc in merged:
            if doc.metadata.get("type") == "image" and doc not in related_images:
                if (doc.metadata.get("source"), doc.metadata.get("page")) in reranked_pages:
                    related_images.append(doc)

        # Combine text + related images for context
        all_context_docs = reranked + related_images[:5]

        # Build context
        context = "\n\n".join([doc.page_content for doc in all_context_docs])

        # Step 5: Sufficiency check
        yield json.dumps({"step": "sufficiency", "status": "running", "detail": "Checking context..."})
        web_docs = []
        sufficient = is_context_sufficient(context, question)
        if sufficient:
            yield json.dumps({"step": "sufficiency", "status": "done", "detail": "Context sufficient ✓"})
        else:
            yield json.dumps({"step": "sufficiency", "status": "done", "detail": "Searching the web..."})
            yield json.dumps({"step": "web_search", "status": "running", "detail": "Web search..."})
            web_docs = web_search(question)
            yield json.dumps({"step": "web_search", "status": "done", "detail": f"{len(web_docs)} web results"})
            if web_docs:
                context += "\n\n[WEB RESULTS]:\n" + "\n\n".join([d.page_content for d in web_docs])

        # Step 6: Generate answer — GPT decides which images to show
        yield json.dumps({"step": "generation", "status": "running", "detail": "Generating answer with GPT-4o..."})
        start = time.time()

        history_ctx = get_history_context()

        prompt = f"""Answer the user's question using ONLY the document context below. Cite page numbers.

DOCUMENT CONTEXT (this is the truth — use this to answer):
{context}

PREVIOUS CONVERSATION (for understanding follow-ups only, do NOT use as factual source):
{history_ctx if history_ctx else "None"}

IMAGE INSTRUCTIONS:
- The context contains entries like "[IMAGE on page X, image #Y]: description..."
- Whenever a Figure or image is mentioned in the context (e.g., "Figure 3", "Figure 4", "as shown in Figure X"), you MUST output: [[SHOW_IMAGE:source_filename|page_number]]
- The source_filename is from the metadata in context. Use the exact filename from the uploads folder, e.g.: ts_usa_investigation_report_public.pdf
- ALWAYS include image tags for every figure referenced in your answer. Do this automatically.
- NEVER say you cannot show images.

FORMAT:
- Use **Markdown** formatting in your answer:
  - Use **bold** for key terms and important facts
  - Use ## headings to organize sections if answer is long
  - Use bullet points or numbered lists for steps/processes
  - Use > blockquotes for direct quotes from documents
  - Use `code style` for technical values (temperatures, measurements, chemical formulas)
  - Use [links](url) for any web sources
  - Add line breaks between paragraphs for readability
- Cite information as **(page X)** in bold italics
- Include ALL relevant details from the context
- If context references Figure 4 on page 15, write: [[SHOW_IMAGE:ts_usa_investigation_report_public.pdf|15]]

Question: {question}

Answer:"""

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        result = llm.invoke(prompt)
        answer_text = result.content
        gen_time = round(time.time() - start, 2)

        # Parse [[SHOW_IMAGE:filename|page]] tags from the answer
        image_pattern = r'\[\[SHOW_IMAGE:(.+?)\|(\d+)\]\]'
        image_refs = re.findall(image_pattern, answer_text)

        # Remove the tags from displayed answer
        clean_answer = re.sub(image_pattern, '', answer_text).strip()

        # Fetch the actual images from the PDFs
        images_to_show = []
        for ref_filename, ref_page in image_refs:
            ref_filename = ref_filename.strip()
            page_int = int(ref_page)
            page_images = extract_page_images(ref_filename, page_int)
            images_to_show.extend(page_images)

        add_to_history("assistant", answer_text)

        # Build sources
        sources = []
        for doc in all_context_docs + web_docs:
            s = {
                "content": doc.page_content[:200] + "...",
                "type": doc.metadata.get("type", "text"),
                "page": doc.metadata.get("page"),
                "source_file": doc.metadata.get("source"),
            }
            if doc.metadata.get("type") == "web":
                s["web_url"] = doc.metadata.get("source")
            sources.append(s)

        yield json.dumps({"step": "sources", "status": "done", "detail": f"{len(sources)} sources", "sources": sources})

        yield json.dumps({
            "step": "generation",
            "status": "done",
            "detail": f"Generated in {gen_time}s",
            "answer": clean_answer,
            "images": images_to_show,
        })

        yield json.dumps({"step": "complete", "status": "done", "detail": "Pipeline complete"})

    return EventSourceResponse(event_stream())


# --- Chat history endpoint ---

@app.get("/history")
async def get_chat_history():
    return {"history": chat_history}


@app.delete("/history")
async def clear_chat_history():
    chat_history.clear()
    return {"message": "History cleared"}


@app.get("/health")
async def health():
    return {"status": "ok"}
