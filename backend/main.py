import os
import json
import shutil
import time
import base64
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
IMAGES_DIR = "extracted_images"
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "documents"
METADATA_FILE = "documents.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

openai_client = OpenAI()


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


def extract_images_from_pdf(file_path: str, filename: str) -> list[Document]:
    """Extract images from PDF and describe them with GPT-4o."""
    doc = fitz.open(file_path)
    image_docs = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Skip very small images (likely icons/logos)
            if len(image_bytes) < 5000:
                continue

            # Save image for reference
            img_filename = f"{filename}_p{page_num + 1}_img{img_idx + 1}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Describe image with GPT-4o
            try:
                description = describe_image(image_bytes, filename, page_num)
                image_docs.append(
                    Document(
                        page_content=f"[IMAGE from page {page_num + 1}]: {description}",
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "type": "image",
                            "image_file": img_filename,
                        },
                    )
                )
            except Exception as e:
                print(f"Failed to describe image {img_filename}: {e}")

    doc.close()
    return image_docs


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
    """SSE endpoint that streams upload/processing progress."""
    global vectorstore

    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save file immediately (before streaming) since file.file is only readable once
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def event_stream():
        global vectorstore

        # Step 1: Saving file
        yield json.dumps({"step": "saving", "status": "done", "detail": f"Saved {filename}"})

        # Step 2: Extracting text
        yield json.dumps({"step": "extracting", "status": "running", "detail": "Extracting text from document..."})
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            yield json.dumps({"step": "error", "status": "done", "detail": "Only .pdf and .txt files are supported"})
            return

        documents = loader.load()
        yield json.dumps({"step": "extracting", "status": "done", "detail": f"Extracted {len(documents)} pages"})

        # Step 3: Chunking
        yield json.dumps({"step": "chunking", "status": "running", "detail": "Splitting into chunks..."})
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        yield json.dumps({"step": "chunking", "status": "done", "detail": f"Created {len(chunks)} text chunks"})

        # Step 4: Image extraction (PDFs only)
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
                                metadata={
                                    "source": filename,
                                    "page": page_num + 1,
                                    "type": "image",
                                    "image_file": img_filename,
                                },
                            )
                        )
                    except Exception as e:
                        print(f"Failed to describe image {img_filename}: {e}")

            doc.close()
            chunks.extend(image_chunks)
            yield json.dumps({"step": "images", "status": "done", "detail": f"Processed {total_images} images"})

        # Step 5: Embedding & storing in Qdrant
        yield json.dumps({"step": "embedding", "status": "running", "detail": f"Embedding {len(chunks)} chunks into Qdrant..."})
        embeddings = OpenAIEmbeddings()
        vectorstore = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            path=QDRANT_PATH,
            collection_name=COLLECTION_NAME,
        )
        yield json.dumps({"step": "embedding", "status": "done", "detail": f"Stored {len(chunks)} chunks in Qdrant"})

        # Save metadata
        docs_meta = load_doc_metadata()
        docs_meta = [d for d in docs_meta if d["filename"] != filename]
        docs_meta.append({
            "filename": filename,
            "chunks": len(chunks),
            "images": len(image_chunks),
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_doc_metadata(docs_meta)

        # Done
        yield json.dumps({"step": "complete", "status": "done", "detail": f"Done! {len(chunks)} chunks indexed ({len(image_chunks)} from images)"})

    return EventSourceResponse(event_stream())


@app.get("/documents")
async def list_documents():
    """Return list of uploaded documents (persists across restarts)."""
    return {"documents": load_doc_metadata()}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and remove it from metadata."""
    global vectorstore

    # Remove file from uploads
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove extracted images for this doc
    if os.path.exists(IMAGES_DIR):
        for img_file in os.listdir(IMAGES_DIR):
            if img_file.startswith(filename):
                os.remove(os.path.join(IMAGES_DIR, img_file))

    # Remove from metadata
    docs_meta = load_doc_metadata()
    docs_meta = [d for d in docs_meta if d["filename"] != filename]
    save_doc_metadata(docs_meta)

    # Reset vectorstore (will be rebuilt on next upload)
    vectorstore = None

    return {"message": f"Deleted {filename}"}


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
        retriever = vs.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(body.question)
        retrieval_time = round(time.time() - start, 2)

        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "type": doc.metadata.get("type", "text"),
            }
            for doc in docs
        ]
        yield json.dumps({
            "step": "retrieval",
            "status": "done",
            "detail": f"Found {len(docs)} relevant chunks in {retrieval_time}s",
            "sources": sources,
        })

        # Step 2: Generating answer with GPT-4o
        yield json.dumps({"step": "generation", "status": "running", "detail": "Sending to GPT-4o..."})

        start = time.time()
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use the following context to answer the question. The context may include descriptions of images, graphs, and charts from documents. If you don't know the answer, say you don't know.

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
