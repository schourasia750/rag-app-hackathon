import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploadSteps, setUploadSteps] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [asking, setAsking] = useState(false);
  const [steps, setSteps] = useState([]);

  useEffect(() => {
    fetchDocuments();
    fetchHistory();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch {}
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/history`);
      const data = await res.json();
      setChatMessages(
        (data.history || []).reduce((acc, msg, i, arr) => {
          if (msg.role === "user") {
            const next = arr[i + 1];
            acc.push({
              question: msg.content,
              answer: next?.role === "assistant" ? next.content : "",
              images: [],
            });
          }
          return acc;
        }, [])
      );
    } catch {}
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadMsg("");
    setUploadSteps([]);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
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
            try {
              const event = JSON.parse(line.slice(6).trim());
              setUploadSteps((prev) => {
                const idx = prev.findIndex((s) => s.step === event.step);
                if (idx >= 0) { const u = [...prev]; u[idx] = event; return u; }
                return [...prev, event];
              });
              if (event.step === "complete") setUploadMsg(event.detail);
            } catch {}
          }
        }
      }
      fetchDocuments();
    } catch (err) {
      setUploadMsg("Error: " + err.message);
    }
    setUploading(false);
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    setAsking(true);
    setSteps([]);

    const currentQuestion = question;
    setQuestion("");

    let currentAnswer = "";
    let currentImages = [];
    let currentSources = [];

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentQuestion }),
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
                if (idx >= 0) { const u = [...prev]; u[idx] = event; return u; }
                return [...prev, event];
              });
              if (event.answer) currentAnswer = event.answer;
              if (event.images && event.images.length > 0) currentImages = event.images;
              if (event.sources) currentSources = event.sources;
            } catch {}
          }
        }
      }
    } catch (err) {
      currentAnswer = "Error: " + err.message;
    }

    setChatMessages((prev) => [
      ...prev,
      { question: currentQuestion, answer: currentAnswer, images: currentImages, sources: currentSources },
    ]);
    setAsking(false);
  };

  const handleDelete = async (filename) => {
    try {
      const res = await fetch(`${API_URL}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (res.ok) fetchDocuments();
    } catch {}
  };

  const clearHistory = async () => {
    await fetch(`${API_URL}/history`, { method: "DELETE" });
    setChatMessages([]);
  };

  return (
    <div className="app">
      <h1>RAG App</h1>

      <section>
        <h2>Upload Document</h2>
        <input type="file" accept=".pdf,.txt" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? "Processing..." : "Upload"}
        </button>
        {uploadSteps.length > 0 && (
          <div className="pipeline">
            <h3>Upload Progress</h3>
            {uploadSteps.map((s, i) => (
              <div key={i} className={`step step-${s.status}`}>
                <span className="step-icon">{s.status === "running" ? "⏳" : "✅"}</span>
                <span className="step-name">{s.step}</span>
                <span className="step-detail">{s.detail}</span>
              </div>
            ))}
          </div>
        )}
        {uploadMsg && <p className="msg">{uploadMsg}</p>}
        {documents.length > 0 && (
          <div className="doc-list">
            <h3>Uploaded Documents</h3>
            {documents.map((doc, i) => (
              <div key={i} className="doc-item">
                <span>📄 {doc.filename}</span>
                <span className="doc-meta">
                  {doc.chunks} chunks • {doc.uploaded_at}
                  <button className="delete-btn" onClick={() => handleDelete(doc.filename)}>✕</button>
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <h2>Ask a Question</h2>

        {/* Chat history */}
        {chatMessages.length > 0 && (
          <div className="chat-history">
            <div className="chat-header">
              <h3>Conversation</h3>
              <button className="clear-btn" onClick={clearHistory}>Clear</button>
            </div>
            {chatMessages.map((msg, i) => (
              <div key={i} className="chat-exchange">
                <div className="chat-user"><strong>You:</strong> {msg.question}</div>
                <div className="chat-assistant">
                  <strong>AI:</strong>
                  <div className="markdown-content">
                    <ReactMarkdown>{msg.answer}</ReactMarkdown>
                  </div>
                  {msg.images && msg.images.length > 0 && (
                    <div className="chat-images">
                      {msg.images.map((img, j) => (
                        <div key={j} className="chat-image-card">
                          <img src={img.url} alt={`Page ${img.page}`} />
                          <span className="image-caption">📄 {img.source} • Page {img.page}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="chat-sources">
                      <strong>Sources:</strong>
                      {msg.sources.map((s, j) => (
                        <span key={j} className="source-tag">
                          {s.type === "web" ? "🌐" : "📄"} {s.source_file || s.web_url} {s.page && `p.${s.page}`}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

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
            <h3>Pipeline</h3>
            {steps.map((s, i) => (
              <div key={i} className={`step step-${s.status}`}>
                <span className="step-icon">{s.status === "running" ? "⏳" : "✅"}</span>
                <span className="step-name">{s.step}</span>
                <span className="step-detail">{s.detail}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
