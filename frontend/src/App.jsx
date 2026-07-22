import { useState, useEffect } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [asking, setAsking] = useState(false);
  const [steps, setSteps] = useState([]);

  // Fetch uploaded documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch {
      // ignore if backend not running yet
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadMsg("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadMsg(data.message);
        fetchDocuments(); // refresh the list
      } else {
        setUploadMsg(data.detail || "Upload failed");
      }
    } catch (err) {
      setUploadMsg("Error: " + err.message);
    }
    setUploading(false);
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    setAsking(true);
    setAnswer("");
    setSteps([]);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const raw = line.slice(6).trim();
            if (!raw) continue;
            try {
              const event = JSON.parse(raw);
              setSteps((prev) => {
                const idx = prev.findIndex((s) => s.step === event.step);
                if (idx >= 0) {
                  const updated = [...prev];
                  updated[idx] = event;
                  return updated;
                }
                return [...prev, event];
              });

              if (event.answer) {
                setAnswer(event.answer);
              }
            } catch {
              // skip malformed
            }
          }
        }
      }
    } catch (err) {
      setAnswer("Error: " + err.message);
    }
    setAsking(false);
  };

  return (
    <div className="app">
      <h1>RAG App</h1>

      <section>
        <h2>Upload Document</h2>
        <input
          type="file"
          accept=".pdf,.txt"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? "Uploading..." : "Upload"}
        </button>
        {uploadMsg && <p className="msg">{uploadMsg}</p>}

        {documents.length > 0 && (
          <div className="doc-list">
            <h3>Uploaded Documents</h3>
            {documents.map((doc, i) => (
              <div key={i} className="doc-item">
                <span>📄 {doc.filename}</span>
                <span className="doc-meta">{doc.chunks} chunks • {doc.uploaded_at}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <h2>Ask a Question</h2>
        <input
          type="text"
          placeholder="Ask something about your documents..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
        />
        <button onClick={handleAsk} disabled={!question.trim() || asking}>
          {asking ? "Thinking..." : "Ask"}
        </button>

        {steps.length > 0 && (
          <div className="pipeline">
            <h3>Pipeline Steps</h3>
            {steps.map((s, i) => (
              <div key={i} className={`step step-${s.status}`}>
                <span className="step-icon">
                  {s.status === "running" ? "⏳" : "✅"}
                </span>
                <span className="step-name">{s.step}</span>
                <span className="step-detail">{s.detail}</span>
                {s.sources && (
                  <div className="sources">
                    {s.sources.map((src, j) => (
                      <div key={j} className="source-chip">
                        📄 {src.content}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {answer && (
          <div className="answer">
            <strong>Answer:</strong>
            <p>{answer}</p>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
