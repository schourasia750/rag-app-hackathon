import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [asking, setAsking] = useState(false);

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
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (res.ok) {
        setAnswer(data.answer);
      } else {
        setAnswer(data.detail || "Error getting answer");
      }
    } catch (err) {
      setAnswer("Error: " + err.message);
    }
    setAsking(false);
  };

  return (
    <div className="app">
      <h1>RAG App</h1>

      <section className="upload-section">
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
      </section>

      <section className="ask-section">
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
