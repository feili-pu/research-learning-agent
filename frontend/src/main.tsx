import { StrictMode, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  DatabaseZap,
  FileText,
  Layers,
  Library,
  Loader2,
  MessageSquareText,
  Microscope,
  RefreshCw,
  Search,
  Upload
} from "lucide-react";
import {
  askQuestion,
  DocumentSummary,
  getDocuments,
  QueryResponse,
  reindexDocuments,
  runStudyTask,
  SourceChunk,
  uploadDocument
} from "./api";
import "./styles.css";

type Mode = "query" | "summary" | "key-points" | "reading-plan";

const modeLabels: Record<Mode, string> = {
  query: "自由问答",
  summary: "资料总览",
  "key-points": "关键点",
  "reading-plan": "阅读计划"
};

function App() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [mode, setMode] = useState<Mode>("summary");
  const [topic, setTopic] = useState("这些文档");
  const [focus, setFocus] = useState("研究主题、方法和核心结论");
  const [question, setQuestion] = useState("这些文档主要讲什么？");
  const [topK, setTopK] = useState(4);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("准备就绪");
  const [error, setError] = useState("");

  const totals = useMemo(() => {
    return documents.reduce(
      (acc, doc) => {
        acc.pages += doc.pages;
        acc.chunks += doc.chunks;
        return acc;
      },
      { pages: 0, chunks: 0 }
    );
  }, [documents]);

  async function refreshDocuments() {
    const docs = await getDocuments();
    setDocuments(docs);
  }

  useEffect(() => {
    refreshDocuments().catch((err) => setError(String(err)));
  }, []);

  async function handleUpload(file: File | null) {
    if (!file) return;
    setBusy(true);
    setError("");
    setStatus(`上传并索引 ${file.name}`);
    try {
      await uploadDocument(file);
      await refreshDocuments();
      setStatus("文档已加入索引");
    } catch (err) {
      setError(String(err));
      setStatus("上传失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleReindex() {
    setBusy(true);
    setError("");
    setStatus("正在重建本地索引");
    try {
      const docs = await reindexDocuments();
      setDocuments(docs);
      setStatus("索引已重建");
    } catch (err) {
      setError(String(err));
      setStatus("重建失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleRun() {
    setBusy(true);
    setError("");
    setStatus(`运行 ${modeLabels[mode]}`);
    try {
      const response =
        mode === "query"
          ? await askQuestion(question, topK)
          : await runStudyTask(mode, topic, focus, topK);
      setResult(response);
      setStatus("已生成回答");
    } catch (err) {
      setError(String(err));
      setStatus("请求失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand-line">
            <Microscope size={22} />
            <span>Research Learning Agent</span>
          </div>
          <h1>论文资料学习台</h1>
        </div>
        <div className="status-pill">
          {busy ? <Loader2 className="spin" size={16} /> : <DatabaseZap size={16} />}
          <span>{status}</span>
        </div>
      </header>

      <section className="metrics-band">
        <Metric icon={<Library size={18} />} label="文档" value={documents.length} />
        <Metric icon={<BookOpen size={18} />} label="页数" value={totals.pages} />
        <Metric icon={<Layers size={18} />} label="Chunks" value={totals.chunks} />
      </section>

      <div className="workspace">
        <aside className="sidebar">
          <section className="panel">
            <div className="panel-heading">
              <h2>资料库</h2>
              <button className="icon-button" onClick={handleReindex} disabled={busy} title="重新索引">
                <RefreshCw size={17} />
              </button>
            </div>
            <label className="upload-zone">
              <Upload size={22} />
              <span>上传 PDF</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(event) => handleUpload(event.target.files?.[0] ?? null)}
              />
            </label>
            <div className="doc-list">
              {documents.map((doc) => (
                <article className="doc-row" key={doc.document_id}>
                  <FileText size={16} />
                  <div>
                    <strong>{doc.filename}</strong>
                    <span>{doc.pages} 页 · {doc.chunks} chunks</span>
                  </div>
                </article>
              ))}
              {!documents.length && <p className="empty-text">还没有索引文档。</p>}
            </div>
          </section>
        </aside>

        <section className="main-panel">
          <div className="mode-tabs">
            {(Object.keys(modeLabels) as Mode[]).map((item) => (
              <button
                key={item}
                className={mode === item ? "active" : ""}
                onClick={() => setMode(item)}
              >
                {item === "query" ? <MessageSquareText size={16} /> : <Search size={16} />}
                {modeLabels[item]}
              </button>
            ))}
          </div>

          <div className="input-grid">
            {mode === "query" ? (
              <label className="field full">
                <span>问题</span>
                <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />
              </label>
            ) : (
              <>
                <label className="field">
                  <span>主题</span>
                  <input value={topic} onChange={(event) => setTopic(event.target.value)} />
                </label>
                <label className="field">
                  <span>关注重点</span>
                  <input value={focus} onChange={(event) => setFocus(event.target.value)} />
                </label>
              </>
            )}
            <label className="field compact">
              <span>检索片段数：{topK}</span>
              <input
                type="range"
                min={1}
                max={12}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>
            <button className="primary-button" onClick={handleRun} disabled={busy}>
              {busy ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
              运行
            </button>
          </div>

          {error && <div className="error-box">{error}</div>}

          <ResultView result={result} />
        </section>
      </div>
    </main>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: number }) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ResultView({ result }: { result: QueryResponse | null }) {
  if (!result) {
    return (
      <section className="result-empty">
        <Microscope size={32} />
        <p>选择任务后运行，答案和引用来源会显示在这里。</p>
      </section>
    );
  }

  return (
    <section className="result-layout">
      <article className="answer-panel">
        <div className="answer-meta">
          <span>{result.retrieval_mode}</span>
          <span>{result.answer_mode}</span>
          <span>{result.model ?? "local"}</span>
        </div>
        <div className="answer-text">{result.answer}</div>
      </article>
      <aside className="sources-panel">
        <h2>来源片段</h2>
        {result.sources.map((source, index) => (
          <SourceItem key={source.chunk_id} source={source} index={index + 1} />
        ))}
      </aside>
    </section>
  );
}

function SourceItem({ source, index }: { source: SourceChunk; index: number }) {
  return (
    <article className="source-card">
      <div className="source-head">
        <strong>[{index}] 第 {source.page} 页</strong>
        <span>{source.score.toFixed(3)}</span>
      </div>
      <p>{source.text}</p>
      <small>{source.filename}</small>
    </article>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
