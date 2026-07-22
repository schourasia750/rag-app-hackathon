# RAG App

A simple Retrieval-Augmented Generation app. Upload PDF/TXT documents, then ask questions about them.

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, FAISS, OpenAI
- **Frontend**: React (Vite)

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` folder:

```
OPENAI_API_KEY=sk-your-key-here
```

Start the server:

```bash
uvicorn main:app --reload
```

Runs on http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on http://localhost:5173

## Usage

1. Upload a `.pdf` or `.txt` file using the upload section
2. Ask questions about the document content
3. Get AI-powered answers based on the document
