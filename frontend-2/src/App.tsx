import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Activity, AlertTriangle, ArrowUp, Boxes, CheckCircle2,
  ChevronRight, CircuitBoard, Cpu, Database, FileText,
  Gauge, Globe, HardHat, Layers, Link2, Loader2, Network,
  Radio, ScrollText, Search, Shield, Sparkles, Trash2,
  Upload, Wrench, X, Zap,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
/* ------------------------------- types ---------------------------------- */
type PipelineStep = { step: string; status: string; detail?: string };
type ImageRef = { url: string; page?: number; source?: string };
type SourceRef = { type?: string; source_file?: string; web_url?: string; page?: number };
type ChatMsg = { question: string; answer: string; images: ImageRef[]; sources: SourceRef[] };
type DocItem = { filename: string; chunks: number; uploaded_at: string };

/* ---------------------------- suggested prompts ------------------------- */
const SUGGESTED = [
  { icon: Wrench, label: "RCA on last vibration alarm", q: "What is the likely root cause of the most recent vibration alarm on the main compressor, based on maintenance history?" },
  { icon: Shield, label: "PESO compliance gaps", q: "List any PESO compliance gaps for LPG storage vessels in the current inspection records." },
  { icon: CircuitBoard, label: "Explain this P&ID loop", q: "Explain the control loop around FIC-201 including interlocks and set points." },
  { icon: HardHat, label: "SOP for pump changeover", q: "Give me the step-by-step SOP for changing over Pump P-101A to P-101B, including permits." },
];

/* --------------------------- side navigation ---------------------------- */
const NAV = [
  { id: "copilot", label: "Copilot", icon: Sparkles, badge: "RAG" },
  { id: "corpus", label: "Corpus", icon: Layers },
  { id: "graph", label: "Knowledge Graph", icon: Network },
  { id: "assets", label: "Asset Registry", icon: Boxes },
  { id: "compliance", label: "Compliance", icon: Shield },
  { id: "insights", label: "Signals", icon: Activity },
] as const;

type NavId = string;

export default function App() {
  const [view, setView] = useState<NavId>("copilot");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploadSteps, setUploadSteps] = useState<PipelineStep[]>([]);
  const [documents, setDocuments] = useState<DocItem[]>([]);
  const [question, setQuestion] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMsg[]>([]);
  const [asking, setAsking] = useState(false);
  const [steps, setSteps] = useState<PipelineStep[]>([]);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => { fetchDocuments(); fetchHistory(); }, []);
  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chatMessages, steps, asking]);

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
        (data.history || []).reduce((acc: ChatMsg[], msg: any, i: number, arr: any[]) => {
          if (msg.role === "user") {
            const next = arr[i + 1];
            acc.push({ question: msg.content, answer: next?.role === "assistant" ? next.content : "", images: [], sources: [] });
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
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
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
    } catch (err: any) { setUploadMsg("Error: " + err.message); }
    setUploading(false);
  };

  const handleAsk = async (overrideQ?: string) => {
    const q = (overrideQ ?? question).trim();
    if (!q) return;
    setAsking(true);
    setSteps([]);
    setQuestion("");
    let currentAnswer = "";
    let currentImages: ImageRef[] = [];
    let currentSources: SourceRef[] = [];
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
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
              if (event.images?.length) currentImages = event.images;
              if (event.sources) currentSources = event.sources;
            } catch {}
          }
        }
      }
    } catch (err: any) { currentAnswer = "Error: " + err.message; }
    setChatMessages((prev) => [...prev, { question: q, answer: currentAnswer, images: currentImages, sources: currentSources }]);
    setAsking(false);
  };

  const handleDelete = async (filename: string) => {
    try {
      const res = await fetch(`${API_URL}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (res.ok) fetchDocuments();
    } catch {}
  };

  const handleReset = async () => {
    if (!confirm("This will delete ALL documents, vectors, and chat history. Continue?")) return;
    try {
      const res = await fetch(`${API_URL}/reset`, { method: "POST" });
      if (res.ok) {
        fetchDocuments();
        setChatMessages([]);
        setSteps([]);
      }
    } catch {}
  };

  const clearHistory = async () => {
    await fetch(`${API_URL}/history`, { method: "DELETE" });
    setChatMessages([]);
    setSteps([]);
  };

  const totalChunks = useMemo(() => documents.reduce((s, d) => s + (d.chunks || 0), 0), [documents]);

  return (
    <div className="min-h-screen w-full bg-background text-foreground flex">
      {/* ============ SIDEBAR ============ */}
      <aside className="w-64 shrink-0 border-r border-border bg-surface flex flex-col">
        <div className="px-5 py-5 border-b border-border">
          <div className="flex items-center gap-2.5">
            <div className="relative h-9 w-9 rounded-md bg-signal/15 border border-signal/40 flex items-center justify-center">
              <CircuitBoard className="h-5 w-5 text-signal" />
              <span className="absolute -bottom-0.5 -right-0.5 h-2 w-2 rounded-full bg-safe pulse-dot" />
            </div>
            <div className="leading-tight">
              <div className="font-display font-bold text-[15px] tracking-tight">FORGE</div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Ops Brain · v0.9</div>
            </div>
          </div>
        </div>
        <nav className="p-3 space-y-0.5 flex-1">
          <div className="px-2 py-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Workbench</div>
          {NAV.map((n) => {
            const Icon = n.icon;
            const active = view === n.id;
            return (
              <button key={n.id} onClick={() => setView(n.id)}
                className={`w-full group flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all relative ${active ? "bg-signal/10 text-foreground border border-signal/30" : "text-muted-foreground hover:bg-surface-2 hover:text-foreground border border-transparent"}`}>
                {active && <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-r bg-signal" />}
                <Icon className={`h-4 w-4 ${active ? "text-signal" : ""}`} />
                <span className="flex-1 text-left">{n.label}</span>
                {"badge" in n && n.badge && <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-schematic/15 text-schematic border border-schematic/30">{n.badge}</span>}
              </button>
            );
          })}
        </nav>
        <div className="p-3 border-t border-border">
          <div className="rounded-md bg-surface-2 border border-border p-3 space-y-2">
            <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
              <Radio className="h-3 w-3 text-safe pulse-dot" /> Corpus Status
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Docs" value={documents.length} />
              <Stat label="Chunks" value={totalChunks} />
            </div>
          </div>
        </div>
      </aside>

      {/* ============ MAIN ============ */}
      <main className="flex-1 min-w-0 flex flex-col">
        <TopBar view={view} />
        {view === "copilot" && <CopilotView chatMessages={chatMessages} steps={steps} asking={asking} question={question} setQuestion={setQuestion} handleAsk={handleAsk} clearHistory={clearHistory} documents={documents} chatScrollRef={chatScrollRef} />}
        {view === "corpus" && <CorpusView file={file} setFile={setFile} uploading={uploading} handleUpload={handleUpload} uploadSteps={uploadSteps} uploadMsg={uploadMsg} documents={documents} handleDelete={handleDelete} handleReset={handleReset} />}
        {view === "graph" && <GraphView documents={documents} />}
        {view === "assets" && <AssetsView />}
        {view === "compliance" && <ComplianceView />}
        {view === "insights" && <InsightsView />}
      </main>
    </div>
  );
}

/* ============================ subcomponents ============================= */

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded bg-background/60 border border-border p-2">
      <div className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="font-display font-semibold text-lg leading-none mt-1">{value}</div>
    </div>
  );
}

function TopBar({ view }: { view: NavId }) {
  const label = NAV.find((n) => n.id === view)?.label ?? "";
  return (
    <div className="h-14 border-b border-border bg-surface/60 backdrop-blur-sm px-6 flex items-center justify-between">
      <div className="flex items-center gap-3 text-sm">
        <span className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">FORGE</span>
        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="font-medium">{label}</span>
      </div>
      <div className="flex items-center gap-4 text-[11px] font-mono text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-safe pulse-dot" />API · localhost:8000</span>
        <span className="flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5" />RAG · Hybrid</span>
        <span className="flex items-center gap-1.5"><Database className="h-3.5 w-3.5" />Vector · Qdrant</span>
      </div>
    </div>
  );
}

/* ------------------------------ Copilot --------------------------------- */

function CopilotView(props: {
  chatMessages: ChatMsg[]; steps: PipelineStep[]; asking: boolean;
  question: string; setQuestion: (s: string) => void; handleAsk: (q?: string) => void;
  clearHistory: () => void; documents: DocItem[]; chatScrollRef: React.RefObject<HTMLDivElement | null>;
}) {
  const { chatMessages, steps, asking, question, setQuestion, handleAsk, clearHistory, documents, chatScrollRef } = props;
  const empty = chatMessages.length === 0 && steps.length === 0;

  return (
    <div className="flex-1 min-h-0 grid grid-cols-[1fr_320px]">
      <div className="min-w-0 flex flex-col">
        <div ref={chatScrollRef} className="flex-1 overflow-y-auto blueprint-grid-fine">
          <div className="max-w-3xl mx-auto px-6 py-8">
            {empty ? <EmptyState onPick={(q) => handleAsk(q)} corpusCount={documents.length} /> : (
              <div className="space-y-8">
                {chatMessages.map((m, i) => <ChatExchange key={i} msg={m} />)}
                {asking && <ThinkingBlock steps={steps} />}
              </div>
            )}
          </div>
        </div>
        {/* Composer */}
        <div className="border-t border-border bg-surface/70 backdrop-blur px-6 py-4">
          <div className="max-w-3xl mx-auto">
            <div className="relative rounded-lg border border-border bg-background focus-within:border-signal/60 focus-within:ring-1 focus-within:ring-signal/30 transition">
              <div className="absolute left-3 top-3 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest text-muted-foreground pointer-events-none">
                <Search className="h-3 w-3" />Query · corpus + web
              </div>
              <textarea value={question} onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); } }}
                placeholder="Ask about equipment, procedures, incidents, drawings…" rows={2}
                className="w-full bg-transparent pl-3 pr-14 pt-8 pb-3 text-sm outline-none resize-none placeholder:text-muted-foreground/60" />
              <button onClick={() => handleAsk()} disabled={!question.trim() || asking}
                className="absolute right-2.5 bottom-2.5 h-9 w-9 rounded-md bg-signal text-signal-foreground hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition">
                {asking ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
              </button>
            </div>
            <div className="flex items-center justify-between mt-2 px-1 text-[10px] font-mono text-muted-foreground">
              <div className="flex gap-3"><span>ENTER · send</span><span>⇧ ENTER · newline</span></div>
              {chatMessages.length > 0 && (
                <button onClick={clearHistory} className="hover:text-hazard flex items-center gap-1 transition">
                  <Trash2 className="h-3 w-3" />Clear session
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
      <aside className="border-l border-border bg-surface/40 overflow-y-auto">
        <ContextPanel documents={documents} lastMsg={chatMessages[chatMessages.length - 1]} />
      </aside>
    </div>
  );
}

function EmptyState({ onPick, corpusCount }: { onPick: (q: string) => void; corpusCount: number }) {
  return (
    <div className="pt-6 pb-4">
      <div className="corner-brackets border border-border rounded-lg p-6 bg-surface/50 relative">
        <div className="flex items-start gap-4">
          <div className="h-12 w-12 rounded-md bg-signal/15 border border-signal/40 flex items-center justify-center shrink-0">
            <Sparkles className="h-6 w-6 text-signal" />
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-signal">Expert Knowledge Copilot</div>
            <h1 className="font-display text-2xl font-semibold mt-1">One brain for every drawing, work order, and SOP.</h1>
            <p className="text-sm text-muted-foreground mt-2 max-w-xl">
              Ask anything across your industrial corpus — P&IDs, maintenance history, inspection reports, safety procedures. Answers come with source citations, page references and cited images.
            </p>
            <div className="mt-3 flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
              <span className="flex items-center gap-1.5"><Layers className="h-3 w-3" /> {corpusCount} documents indexed</span>
              <span className="h-3 w-px bg-border" />
              <span className="flex items-center gap-1.5"><Globe className="h-3 w-3" /> web fallback ready</span>
            </div>
          </div>
        </div>
      </div>
      <div className="mt-6">
        <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground mb-2 px-1">Suggested queries</div>
        <div className="grid grid-cols-2 gap-2">
          {SUGGESTED.map((s, i) => {
            const Icon = s.icon;
            return (
              <button key={i} onClick={() => onPick(s.q)} className="group text-left rounded-md border border-border bg-surface hover:bg-surface-2 hover:border-signal/40 p-3 transition">
                <div className="flex items-center gap-2 text-signal"><Icon className="h-4 w-4" /><span className="text-xs font-mono uppercase tracking-wide">{s.label}</span></div>
                <div className="text-sm mt-1.5 text-foreground/90 leading-snug">{s.q}</div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ChatExchange({ msg }: { msg: ChatMsg }) {
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-3">
        <div className="h-7 w-7 rounded-md bg-schematic/15 border border-schematic/40 flex items-center justify-center shrink-0">
          <HardHat className="h-4 w-4 text-schematic" />
        </div>
        <div className="flex-1 pt-0.5">
          <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Operator</div>
          <div className="text-[15px] font-medium mt-0.5">{msg.question}</div>
        </div>
      </div>
      <div className="flex items-start gap-3">
        <div className="h-7 w-7 rounded-md bg-signal/15 border border-signal/40 flex items-center justify-center shrink-0">
          <Sparkles className="h-4 w-4 text-signal" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Forge · answer</div>
          <div className="md-body mt-1"><ReactMarkdown>{msg.answer}</ReactMarkdown></div>
          {msg.images?.length > 0 && (
            <div className="mt-4">
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground mb-2">Cited figures</div>
              <div className="grid grid-cols-2 gap-2">
                {msg.images.map((img, j) => (
                  <div key={j} className="rounded-md border border-border bg-surface overflow-hidden group">
                    <div className="aspect-video bg-background/60 overflow-hidden">
                      <img src={img.url} alt={`Page ${img.page}`} className="w-full h-full object-contain group-hover:scale-[1.02] transition" />
                    </div>
                    <div className="px-2.5 py-1.5 text-[10px] font-mono flex items-center gap-1.5 text-muted-foreground border-t border-border">
                      <FileText className="h-3 w-3" /><span className="truncate">{img.source}</span>
                      {img.page != null && <span className="text-signal">· p.{img.page}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {msg.sources?.length > 0 && (
            <div className="mt-4">
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground mb-2">Sources</div>
              <div className="flex flex-wrap gap-1.5">
                {msg.sources.map((s, j) => (
                  <span key={j} className="inline-flex items-center gap-1.5 text-[11px] font-mono px-2 py-1 rounded bg-surface border border-border hover:border-signal/40 transition">
                    {s.type === "web" ? <Globe className="h-3 w-3 text-schematic" /> : <FileText className="h-3 w-3 text-signal" />}
                    <span className="truncate max-w-[240px]">{s.source_file || s.web_url}</span>
                    {s.page != null && <span className="text-muted-foreground">p.{s.page}</span>}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ThinkingBlock({ steps }: { steps: PipelineStep[] }) {
  return (
    <div className="flex items-start gap-3">
      <div className="h-7 w-7 rounded-md bg-signal/15 border border-signal/40 flex items-center justify-center shrink-0 scan-line">
        <Loader2 className="h-4 w-4 text-signal animate-spin" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[10px] font-mono uppercase tracking-widest text-signal">Reasoning pipeline</div>
        <div className="mt-2 rounded-md border border-border bg-surface/70 p-3 space-y-1.5">
          {steps.map((s, i) => <PipelineRow key={i} step={s} />)}
          {steps.length === 0 && <div className="text-xs font-mono text-muted-foreground">Initializing…</div>}
        </div>
      </div>
    </div>
  );
}

function PipelineRow({ step }: { step: PipelineStep }) {
  const running = step.status === "running";
  const done = step.status === "complete" || step.status === "done" || step.status === "success";
  const err = step.status === "error" || step.status === "failed";
  return (
    <div className="flex items-start gap-2.5 text-[12px] font-mono">
      <span className="mt-0.5 shrink-0">
        {running && <Loader2 className="h-3.5 w-3.5 text-signal animate-spin" />}
        {done && <CheckCircle2 className="h-3.5 w-3.5 text-safe" />}
        {err && <AlertTriangle className="h-3.5 w-3.5 text-hazard" />}
        {!running && !done && !err && <span className="block h-3.5 w-3.5 rounded-full border border-border" />}
      </span>
      <span className={`uppercase tracking-wide shrink-0 min-w-[110px] ${done ? "text-safe" : running ? "text-signal" : err ? "text-hazard" : "text-muted-foreground"}`}>{step.step}</span>
      {step.detail && <span className="text-muted-foreground truncate">{step.detail}</span>}
    </div>
  );
}

function ContextPanel({ documents, lastMsg }: { documents: DocItem[]; lastMsg?: ChatMsg }) {
  return (
    <div className="p-4 space-y-4">
      <SectionLabel icon={Gauge} text="Session Telemetry" />
      <div className="grid grid-cols-3 gap-2">
        <MiniStat label="Docs" value={documents.length} tone="signal" />
        <MiniStat label="Cites" value={lastMsg?.sources?.length ?? 0} tone="schematic" />
        <MiniStat label="Figs" value={lastMsg?.images?.length ?? 0} tone="safe" />
      </div>
      <SectionLabel icon={Layers} text="Active Corpus" />
      <div className="space-y-1.5 max-h-[280px] overflow-y-auto pr-1">
        {documents.length === 0 && <div className="text-xs text-muted-foreground font-mono">No documents yet · upload from Corpus tab</div>}
        {documents.map((d, i) => (
          <div key={i} className="rounded border border-border bg-background/50 p-2 text-xs flex items-center gap-2">
            <FileText className="h-3.5 w-3.5 text-signal shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">{d.filename}</div>
              <div className="text-[10px] font-mono text-muted-foreground">{d.chunks} chunks</div>
            </div>
          </div>
        ))}
      </div>
      <SectionLabel icon={Zap} text="Live Signals" />
      <div className="space-y-1.5">
        {[
          { l: "Vibration · MC-101", v: "4.2 mm/s", tone: "safe" as const },
          { l: "Temp · HX-204", v: "128 °C", tone: "signal" as const },
          { l: "Δ Pressure · F-11", v: "0.42 bar", tone: "schematic" as const },
          { l: "Permits open", v: "3", tone: "hazard" as const },
        ].map((r, i) => (
          <div key={i} className="flex items-center justify-between text-[11px] font-mono rounded border border-border bg-background/50 px-2 py-1.5">
            <span className="text-muted-foreground">{r.l}</span>
            <span className={r.tone === "safe" ? "text-safe" : r.tone === "signal" ? "text-signal" : r.tone === "hazard" ? "text-hazard" : "text-schematic"}>{r.v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionLabel({ icon: Icon, text }: { icon: any; text: string }) {
  return (
    <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
      <Icon className="h-3 w-3" />{text}<span className="flex-1 h-px bg-border ml-1" />
    </div>
  );
}

function MiniStat({ label, value, tone }: { label: string; value: number | string; tone: "signal" | "schematic" | "safe" }) {
  const color = tone === "signal" ? "text-signal" : tone === "schematic" ? "text-schematic" : "text-safe";
  return (
    <div className="rounded border border-border bg-background/50 p-2">
      <div className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className={`font-display font-semibold text-base leading-none mt-1 ${color}`}>{value}</div>
    </div>
  );
}

/* ------------------------------ Corpus ---------------------------------- */

function CorpusView(props: {
  file: File | null; setFile: (f: File | null) => void; uploading: boolean;
  handleUpload: () => void; uploadSteps: PipelineStep[]; uploadMsg: string;
  documents: DocItem[]; handleDelete: (f: string) => void; handleReset: () => void;
}) {
  const { file, setFile, uploading, handleUpload, uploadSteps, uploadMsg, documents, handleDelete, handleReset } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  return (
    <div className="flex-1 overflow-y-auto blueprint-grid-fine">
      <div className="max-w-5xl mx-auto p-8 space-y-8">
        <header>
          <div className="text-[10px] font-mono uppercase tracking-widest text-signal">Universal Document Ingestion</div>
          <h1 className="font-display text-2xl font-semibold mt-1">Corpus</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Drop PDFs, drawings, SOPs and inspection scans. FORGE extracts entities, builds embeddings, and indexes them into the knowledge graph in real time.
          </p>
        </header>

        {/* Uploader */}
        <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) setFile(f); }}
          className={`corner-brackets rounded-lg border-2 border-dashed p-8 transition ${drag ? "border-signal bg-signal/5" : "border-border bg-surface/40"}`}>
          <input ref={inputRef} type="file" accept=".pdf,.txt" className="hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <div className="flex items-center gap-5">
            <div className="h-14 w-14 rounded-md bg-signal/15 border border-signal/40 flex items-center justify-center shrink-0">
              <Upload className="h-6 w-6 text-signal" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-display text-lg font-semibold">{file ? file.name : "Drop a document or click to browse"}</div>
              <div className="text-xs font-mono text-muted-foreground mt-1">
                Accepted · PDF · TXT · {file ? `${(file.size / 1024).toFixed(1)} KB` : "P&IDs, work orders, SOPs, inspection reports"}
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => inputRef.current?.click()} className="px-4 py-2 rounded-md border border-border bg-surface hover:bg-surface-2 text-sm font-medium transition">Browse</button>
              <button onClick={handleUpload} disabled={!file || uploading}
                className="px-4 py-2 rounded-md bg-signal text-signal-foreground hover:brightness-110 disabled:opacity-40 text-sm font-medium flex items-center gap-2 transition">
                {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                {uploading ? "Ingesting…" : "Ingest"}
              </button>
            </div>
          </div>
          {uploadSteps.length > 0 && (
            <div className="mt-6 rounded-md border border-border bg-background/60 p-4 space-y-1.5">
              <div className="text-[10px] font-mono uppercase tracking-widest text-signal mb-2">Ingestion Pipeline</div>
              {uploadSteps.map((s, i) => <PipelineRow key={i} step={s} />)}
            </div>
          )}
          {uploadMsg && <div className="mt-3 text-xs font-mono text-safe flex items-center gap-2"><CheckCircle2 className="h-4 w-4" /> {uploadMsg}</div>}
        </div>

        {/* Document list */}
        <div>
          <div className="flex items-baseline justify-between mb-3">
            <div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Indexed</div>
              <div className="font-display text-lg font-semibold">{documents.length} document{documents.length === 1 ? "" : "s"} · corpus</div>
            </div>
            {documents.length > 0 && (
              <button onClick={handleReset} className="ml-auto px-3 py-1.5 text-xs font-mono rounded border border-hazard/50 text-hazard hover:bg-hazard/10 transition flex items-center gap-1.5">
                <Trash2 className="h-3 w-3" /> Reset All
              </button>
            )}
          </div>
          {documents.length === 0 ? (
            <div className="rounded-md border border-dashed border-border p-10 text-center">
              <ScrollText className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <div className="text-sm text-muted-foreground">No documents indexed yet. Upload one above to bootstrap the knowledge graph.</div>
            </div>
          ) : (
            <div className="grid gap-2">
              {documents.map((d, i) => (
                <div key={i} className="group grid grid-cols-[auto_1fr_auto_auto] items-center gap-4 rounded-md border border-border bg-surface hover:bg-surface-2 hover:border-signal/40 px-4 py-3 transition">
                  <div className="h-10 w-10 rounded bg-signal/10 border border-signal/30 flex items-center justify-center">
                    <FileText className="h-5 w-5 text-signal" />
                  </div>
                  <div className="min-w-0">
                    <div className="font-medium truncate">{d.filename}</div>
                    <div className="text-[11px] font-mono text-muted-foreground mt-0.5">Ingested · {d.uploaded_at}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono px-2 py-1 rounded bg-schematic/10 border border-schematic/30 text-schematic">{d.chunks} chunks</span>
                    <span className="text-[10px] font-mono px-2 py-1 rounded bg-safe/10 border border-safe/30 text-safe flex items-center gap-1"><CheckCircle2 className="h-3 w-3" /> indexed</span>
                  </div>
                  <button onClick={() => handleDelete(d.filename)} className="h-8 w-8 rounded flex items-center justify-center text-muted-foreground hover:text-hazard hover:bg-hazard/10 transition" title="Remove">
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------- Knowledge Graph ----------------------------- */

function GraphView({ documents }: { documents: DocItem[] }) {
  const nodes = useMemo(() => [
    { l: "MC-101", g: "asset" }, { l: "FIC-201", g: "instrument" },
    { l: "P-101A/B", g: "asset" }, { l: "SOP-045", g: "procedure" },
    { l: "OISD-116", g: "regulation" }, { l: "WO-2874", g: "workorder" },
    { l: "INC-2025-03", g: "incident" }, { l: "HX-204", g: "asset" },
  ], []);

  const groups: Record<string, string> = {
    asset: "text-signal border-signal/40 bg-signal/10",
    instrument: "text-schematic border-schematic/40 bg-schematic/10",
    procedure: "text-safe border-safe/40 bg-safe/10",
    regulation: "text-hazard border-hazard/40 bg-hazard/10",
    workorder: "text-foreground border-border bg-surface-2",
    incident: "text-hazard border-hazard/40 bg-hazard/10",
  };

  return (
    <div className="flex-1 overflow-y-auto blueprint-grid">
      <div className="max-w-6xl mx-auto p-8 space-y-6">
        <header>
          <div className="text-[10px] font-mono uppercase tracking-widest text-signal">Industrial Ontology · Knowledge Graph</div>
          <h1 className="font-display text-2xl font-semibold mt-1">Entity Web</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Entities extracted across your corpus — equipment tags, instruments, procedures, regulations, incidents — linked by co-occurrence and semantic relationship.
          </p>
        </header>
        <div className="relative rounded-lg border border-border bg-surface/40 aspect-[16/9] overflow-hidden">
          <svg className="absolute inset-0 w-full h-full">
            {[[15,30,50,45],[50,45,82,25],[50,45,78,68],[50,45,22,72],[22,72,45,82],[78,68,60,85],[50,45,60,15]].map((c, i) => (
              <line key={i} x1={`${c[0]}%`} y1={`${c[1]}%`} x2={`${c[2]}%`} y2={`${c[3]}%`} stroke="currentColor" strokeWidth="1" strokeDasharray="3 4" className="text-schematic/50" />
            ))}
          </svg>
          {nodes.map((n, i) => {
            const positions = [[15,30],[50,45],[82,25],[78,68],[22,72],[60,15],[45,82],[60,85]];
            const [x, y] = positions[i % positions.length];
            return <div key={i} style={{ left: `${x}%`, top: `${y}%` }} className={`absolute -translate-x-1/2 -translate-y-1/2 px-2.5 py-1 rounded border font-mono text-[11px] ${groups[n.g]} backdrop-blur-sm`}>{n.l}</div>;
          })}
        </div>
        <div className="grid grid-cols-4 gap-3">
          {[{ l: "Assets", v: 42, i: Boxes },{ l: "Procedures", v: 118, i: ScrollText },{ l: "Incidents", v: 9, i: AlertTriangle },{ l: "Relations", v: 1284, i: Link2 }].map((c, i) => {
            const Icon = c.i;
            return (
              <div key={i} className="rounded-md border border-border bg-surface p-4">
                <div className="flex items-center justify-between text-muted-foreground"><div className="text-[10px] font-mono uppercase tracking-widest">{c.l}</div><Icon className="h-4 w-4" /></div>
                <div className="font-display text-2xl font-semibold mt-2">{c.v.toLocaleString()}</div>
              </div>
            );
          })}
        </div>
        <div className="text-[10px] font-mono text-muted-foreground">Graph derived from {documents.length} live document{documents.length === 1 ? "" : "s"} · demonstration overlay</div>
      </div>
    </div>
  );
}

/* ------------------------------ Assets ---------------------------------- */

function AssetsView() {
  const assets = [
    { tag: "MC-101", name: "Main Compressor", health: 92, status: "safe", next: "Q3 · overhaul" },
    { tag: "HX-204", name: "Feed/Effluent Exchanger", health: 74, status: "signal", next: "Nov 12 · cleaning" },
    { tag: "P-101A", name: "Reflux Pump A", health: 61, status: "signal", next: "Oct 30 · seal insp." },
    { tag: "V-311", name: "Knock-out Drum", health: 88, status: "safe", next: "2026 · UT scan" },
    { tag: "F-11", name: "Cracker Furnace", health: 44, status: "hazard", next: "Immediate · tube leak" },
    { tag: "T-812", name: "Storage Tank", health: 79, status: "safe", next: "Feb · API 653" },
  ];
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-6xl mx-auto p-8 space-y-6">
        <header>
          <div className="text-[10px] font-mono uppercase tracking-widest text-signal">Maintenance Intelligence</div>
          <h1 className="font-display text-2xl font-semibold mt-1">Asset Registry</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Live health index fused from work order history, inspection findings, and OEM manuals — updated as new records are ingested.
          </p>
        </header>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {assets.map((a, i) => (
            <div key={i} className="rounded-md border border-border bg-surface p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-mono text-[11px] text-schematic">{a.tag}</div>
                  <div className="font-display font-semibold mt-0.5">{a.name}</div>
                </div>
                <span className={`text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border ${a.status === "safe" ? "text-safe border-safe/40 bg-safe/10" : a.status === "signal" ? "text-signal border-signal/40 bg-signal/10" : "text-hazard border-hazard/40 bg-hazard/10"}`}>
                  {a.status === "safe" ? "OK" : a.status === "signal" ? "Watch" : "Action"}
                </span>
              </div>
              <div className="mt-4">
                <div className="flex justify-between text-[10px] font-mono text-muted-foreground mb-1"><span>Health</span><span>{a.health}%</span></div>
                <div className="h-1.5 rounded bg-background overflow-hidden">
                  <div className={`h-full ${a.health > 80 ? "bg-safe" : a.health > 55 ? "bg-signal" : "bg-hazard"}`} style={{ width: `${a.health}%` }} />
                </div>
              </div>
              <div className="mt-3 text-[11px] font-mono text-muted-foreground flex items-center gap-1.5"><Wrench className="h-3 w-3" /> {a.next}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Compliance ------------------------------ */

function ComplianceView() {
  const items = [
    { code: "OISD-116", title: "Fire Protection · Hydrocarbon Processing", status: "pass", cov: 96 },
    { code: "PESO Rule 33", title: "LPG Storage Vessel Inspection", status: "gap", cov: 71 },
    { code: "Factory Act §21", title: "Fencing of Machinery", status: "pass", cov: 100 },
    { code: "IS 14489", title: "HAZOP · Process Safety", status: "review", cov: 84 },
    { code: "CPCB EPR", title: "Effluent Discharge Reporting", status: "pass", cov: 92 },
    { code: "OISD-144", title: "Liquid Hydrocarbon Storage", status: "gap", cov: 63 },
  ];
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-6xl mx-auto p-8 space-y-6">
        <header>
          <div className="text-[10px] font-mono uppercase tracking-widest text-signal">Regulatory Intelligence</div>
          <h1 className="font-display text-2xl font-semibold mt-1">Compliance Matrix</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Automatic mapping of regulatory clauses against current procedures, equipment state and inspection records. Gaps generate work orders; passes generate audit packs.
          </p>
        </header>
        <div className="rounded-md border border-border overflow-hidden">
          <div className="grid grid-cols-[120px_1fr_100px_120px_60px] text-[10px] font-mono uppercase tracking-widest text-muted-foreground bg-surface px-4 py-2 border-b border-border">
            <span>Code</span><span>Clause</span><span>Status</span><span>Coverage</span><span className="text-right">Pack</span>
          </div>
          {items.map((it, i) => (
            <div key={i} className="grid grid-cols-[120px_1fr_100px_120px_60px] items-center px-4 py-3 border-b border-border last:border-0 bg-surface/60 hover:bg-surface-2 text-sm">
              <span className="font-mono text-schematic text-xs">{it.code}</span>
              <span>{it.title}</span>
              <span>
                <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded border ${it.status === "pass" ? "text-safe border-safe/40 bg-safe/10" : it.status === "gap" ? "text-hazard border-hazard/40 bg-hazard/10" : "text-signal border-signal/40 bg-signal/10"}`}>{it.status}</span>
              </span>
              <span className="flex items-center gap-2">
                <div className="h-1 flex-1 rounded bg-background overflow-hidden">
                  <div className={`h-full ${it.cov > 90 ? "bg-safe" : it.cov > 75 ? "bg-signal" : "bg-hazard"}`} style={{ width: `${it.cov}%` }} />
                </div>
                <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{it.cov}</span>
              </span>
              <button className="justify-self-end h-7 w-7 rounded flex items-center justify-center border border-border hover:border-signal/40 hover:text-signal transition">
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Insights -------------------------------- */

function InsightsView() {
  const signals = [
    { t: "T-812 · rim seal wear pattern matches 2023 incident cluster", tone: "hazard", tag: "Lessons Learned" },
    { t: "3 open work orders on HX-204 exceed mean cycle by 42%", tone: "signal", tag: "Maintenance" },
    { t: "PESO Rule 33 gap flagged on 2 LPG bullets — evidence pack drafted", tone: "hazard", tag: "Compliance" },
    { t: "SOP-045 revised → 12 procedures reference old set point", tone: "signal", tag: "Procedure Drift" },
    { t: "Near-miss NM-2025-11 similar to NM-2019-04 · notify shift lead", tone: "safe", tag: "Failure Intel" },
  ];
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto p-8 space-y-6">
        <header>
          <div className="text-[10px] font-mono uppercase tracking-widest text-signal">Lessons Learned & Failure Intelligence</div>
          <h1 className="font-display text-2xl font-semibold mt-1">Signals</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Cross-corpus patterns FORGE surfaces proactively — the kind of connections no single reviewer can make alone.
          </p>
        </header>
        <div className="space-y-2">
          {signals.map((s, i) => {
            const tone = s.tone === "hazard" ? "border-hazard/40 bg-hazard/5" : s.tone === "signal" ? "border-signal/40 bg-signal/5" : "border-safe/40 bg-safe/5";
            const dot = s.tone === "hazard" ? "bg-hazard" : s.tone === "signal" ? "bg-signal" : "bg-safe";
            return (
              <div key={i} className={`rounded-md border ${tone} p-4 flex items-start gap-3`}>
                <span className={`mt-1.5 h-2 w-2 rounded-full ${dot} pulse-dot shrink-0`} />
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">{s.tag}</div>
                  <div className="text-sm mt-0.5">{s.t}</div>
                </div>
                <button className="text-[10px] font-mono uppercase px-2 py-1 rounded border border-border hover:border-signal/40 hover:text-signal transition">Open</button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
