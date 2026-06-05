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
  Network,
  RefreshCw,
  Search,
  Upload
} from "lucide-react";
import {
  askQuestion,
  DocumentSummary,
  getDocuments,
  LiteratureReviewResponse,
  LiteratureSearchResponse,
  PaperCandidate,
  QueryResponse,
  reindexDocuments,
  runLiteratureTask,
  searchLiterature,
  SourceChunk,
  uploadDocument
} from "./api";
import "./styles.css";

type Mode = "direction-review" | "method-map" | "detail-briefing" | "paper-search" | "query";
type AppResult = QueryResponse | LiteratureReviewResponse | LiteratureSearchResponse | null;

const modeLabels: Record<Mode, string> = {
  "direction-review": "方向综述",
  "method-map": "方法梳理",
  "detail-briefing": "细节分析",
  "paper-search": "相关论文",
  query: "自由问答"
};

const modeDescriptions: Record<Mode, string> = {
  "direction-review": "从论文库中找出相关论文，整理研究背景、问题、方法和结论。",
  "method-map": "专门梳理一个方向里的方法类别、技术细节、优缺点和来源论文。",
  "detail-briefing": "围绕关注点展开细节，适合开题、复现或做文献综述前使用。",
  "paper-search": "只返回相关论文排序和证据片段，不生成长回答。",
  query: "直接向当前论文库提问，适合已有明确问题时使用。"
};

function App() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [mode, setMode] = useState<Mode>("direction-review");
  const [direction, setDirection] = useState("图神经网络在推荐系统中的应用");
  const [focus, setFocus] = useState("研究问题、代表方法、实验设置和可复现切入点");
  const [question, setQuestion] = useState("这个方向有哪些代表性方法？");
  const [topKDocuments, setTopKDocuments] = useState(5);
  const [evidenceK, setEvidenceK] = useState(18);
  const [result, setResult] = useState<AppResult>(null);
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
      setStatus("文档已加入论文库");
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
    setStatus("正在重建论文库索引");
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
      const response = await runCurrentTask();
      setResult(response);
      setStatus("已生成结果");
    } catch (err) {
      setError(String(err));
      setStatus("请求失败");
    } finally {
      setBusy(false);
    }
  }

  function runCurrentTask() {
    if (mode === "query") {
      return askQuestion(question, Math.min(evidenceK, 10));
    }
    if (mode === "paper-search") {
      return searchLiterature(direction, focus, topKDocuments, evidenceK);
    }
    const task = mode === "direction-review" ? "review" : mode === "method-map" ? "methods" : "details";
    return runLiteratureTask(task, direction, focus, topKDocuments, evidenceK);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand-line">
            <Microscope size={22} />
            <span>Research Learning Agent</span>
          </div>
          <h1>方向级文献研究台</h1>
        </div>
        <div className="status-pill">
          {busy ? <Loader2 className="spin" size={16} /> : <DatabaseZap size={16} />}
          <span>{status}</span>
        </div>
      </header>

      <section className="metrics-band">
        <Metric icon={<Library size={18} />} label="论文" value={documents.length} />
        <Metric icon={<BookOpen size={18} />} label="页数" value={totals.pages} />
        <Metric icon={<Layers size={18} />} label="证据块" value={totals.chunks} />
      </section>

      <div className="workspace">
        <aside className="sidebar">
          <section className="panel">
            <div className="panel-heading">
              <h2>后台论文库</h2>
              <button className="icon-button" onClick={handleReindex} disabled={busy} title="重新索引">
                <RefreshCw size={17} />
              </button>
            </div>
            <label className="upload-zone">
              <Upload size={22} />
              <span>上传 PDF 到论文库</span>
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
                    <strong>{displayTitle(doc)}</strong>
                    <span>{metadataLine(doc.metadata)} · {doc.pages} 页 · {doc.chunks} chunks</span>
                    <small>{doc.filename}</small>
                    {doc.metadata.duplicate_of && <em>疑似重复：{doc.metadata.duplicate_reason}</em>}
                  </div>
                </article>
              ))}
              {!documents.length && <p className="empty-text">还没有索引论文。</p>}
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
                title={modeDescriptions[item]}
              >
                {item === "query" ? <MessageSquareText size={16} /> : <Network size={16} />}
                {modeLabels[item]}
              </button>
            ))}
          </div>

          <p className="mode-note">{modeDescriptions[mode]}</p>

          <div className="input-grid research-grid">
            {mode === "query" ? (
              <label className="field full">
                <span>问题</span>
                <textarea value={question} onChange={(event) => setQuestion(event.target.value)} />
              </label>
            ) : (
              <>
                <label className="field">
                  <span>研究方向</span>
                  <input value={direction} onChange={(event) => setDirection(event.target.value)} />
                </label>
                <label className="field">
                  <span>关注重点</span>
                  <input value={focus} onChange={(event) => setFocus(event.target.value)} />
                </label>
              </>
            )}
            <label className="field compact">
              <span>候选论文：{topKDocuments}</span>
              <input
                type="range"
                min={1}
                max={10}
                value={topKDocuments}
                disabled={mode === "query"}
                onChange={(event) => setTopKDocuments(Number(event.target.value))}
              />
            </label>
            <label className="field compact">
              <span>证据片段：{mode === "query" ? Math.min(evidenceK, 10) : evidenceK}</span>
              <input
                type="range"
                min={3}
                max={40}
                value={evidenceK}
                onChange={(event) => setEvidenceK(Number(event.target.value))}
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

function ResultView({ result }: { result: AppResult }) {
  if (!result) {
    return (
      <section className="result-empty">
        <Microscope size={32} />
        <p>输入研究方向后运行，系统会先找相关论文，再整理答案和证据。</p>
      </section>
    );
  }

  const papers = "papers" in result ? result.papers : [];
  const hasAnswer = "answer" in result;

  return (
    <section className="result-stack">
      {papers.length > 0 && <PaperList papers={papers} />}
      <div className="result-layout">
        <article className="answer-panel">
          <div className="answer-meta">
            <span>{result.retrieval_mode}</span>
            {hasAnswer && <span>{result.answer_mode}</span>}
            {hasAnswer && <span>{result.model ?? "local"}</span>}
            {"task" in result && <span>{result.task}</span>}
          </div>
          <div className="answer-text">
            {hasAnswer ? result.answer : "已完成相关论文检索。右侧是证据片段，上方是论文级排序。"}
          </div>
        </article>
        <aside className="sources-panel">
          <h2>证据片段</h2>
          {result.sources.map((source, index) => (
            <SourceItem key={source.chunk_id} source={source} index={index + 1} />
          ))}
        </aside>
      </div>
    </section>
  );
}

function PaperList({ papers }: { papers: PaperCandidate[] }) {
  return (
    <section className="paper-panel">
      <div className="panel-heading">
        <h2>相关论文排序</h2>
        <span className="subtle-count">{papers.length} 篇</span>
      </div>
      <div className="paper-grid">
        {papers.map((paper, index) => (
          <article className="paper-card" key={paper.document_id}>
            <div className="paper-rank">{index + 1}</div>
            <div>
              <strong>{displayTitle(paper)}</strong>
              <span>
                {metadataLine(paper.metadata)} · score {paper.score.toFixed(3)} · {paper.evidence_count} 个证据 · 第{" "}
                {paper.evidence_pages.join(", ")} 页
              </span>
              {paper.metadata.doi && <span>DOI {paper.metadata.doi}</span>}
              {paper.metadata.duplicate_of && <em>疑似重复：{paper.metadata.duplicate_reason}</em>}
              <p>{paper.metadata.abstract || paper.preview}</p>
              <small>{paper.filename}</small>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function SourceItem({ source, index }: { source: SourceChunk; index: number }) {
  return (
    <article className="source-card">
      <div className="source-head">
        <strong>
          [{index}] {source.filename} · 第 {source.page} 页
        </strong>
        <span>{source.score.toFixed(3)}</span>
      </div>
      <p>{source.text}</p>
      <small>{source.chunk_id}</small>
    </article>
  );
}

function displayTitle(item: { filename: string; metadata: { title: string | null } }) {
  return item.metadata.title || item.filename;
}

function metadataLine(metadata: {
  authors: string | null;
  year: number | null;
  venue: string | null;
  keywords: string[];
}) {
  const parts = [
    metadata.year ? String(metadata.year) : null,
    metadata.venue,
    metadata.authors
  ].filter(Boolean);
  if (parts.length) {
    return parts.join(" · ");
  }
  if (metadata.keywords.length) {
    return metadata.keywords.slice(0, 3).join(" · ");
  }
  return "metadata pending";
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
