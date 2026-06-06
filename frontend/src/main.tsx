import { StrictMode, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  Compass,
  DatabaseZap,
  FileSearch,
  FileText,
  Layers,
  Library,
  Loader2,
  MessageSquareText,
  Microscope,
  Network,
  Plus,
  RefreshCw,
  Search,
  Upload
} from "lucide-react";
import {
  askQuestion,
  DiscoveryPaper,
  DiscoveryResponse,
  DocumentFilters,
  DocumentSummary,
  enrichMetadata,
  getDocuments,
  importDiscoveredPaper,
  LiteratureReviewResponse,
  LiteratureSearchResponse,
  LiteratureEvaluationResponse,
  PaperCandidate,
  QueryResponse,
  reindexDocuments,
  runLiteratureEvaluation,
  runLiteratureTask,
  searchDiscovery,
  searchLiterature,
  SourceChunk,
  uploadDocument
} from "./api";
import "./styles.css";

type Mode = "discovery" | "direction-review" | "method-map" | "detail-briefing" | "paper-compare" | "paper-search" | "evaluation" | "query";
type AppResult = QueryResponse | LiteratureReviewResponse | LiteratureSearchResponse | LiteratureEvaluationResponse | DiscoveryResponse | null;

const sectionOptions = [
  { value: "", label: "全部章节" },
  { value: "abstract", label: "摘要" },
  { value: "introduction", label: "引言" },
  { value: "related_work", label: "相关工作" },
  { value: "methods", label: "方法" },
  { value: "experiments", label: "实验" },
  { value: "results", label: "结果" },
  { value: "discussion", label: "讨论" },
  { value: "conclusion", label: "结论" }
];

const sectionLabels: Record<string, string> = {
  abstract: "摘要",
  introduction: "引言",
  related_work: "相关工作",
  methods: "方法",
  experiments: "实验",
  results: "结果",
  discussion: "讨论",
  conclusion: "结论",
  references: "参考文献",
  unknown: "未知章节"
};

const modeLabels: Record<Mode, string> = {
  discovery: "文献发现",
  "direction-review": "方向综述",
  "method-map": "方法梳理",
  "detail-briefing": "细节分析",
  "paper-compare": "论文对比",
  "paper-search": "相关论文",
  evaluation: "检索评估",
  query: "自由问答"
};

const modeDescriptions: Record<Mode, string> = {
  discovery: "从 Semantic Scholar、Crossref、arXiv 和 OpenAlex 搜索候选论文，并可先导入元数据。",
  "direction-review": "从论文库中找出相关论文，整理研究背景、问题、方法和结论。",
  "method-map": "专门梳理一个方向里的方法类别、技术细节、优缺点和来源论文。",
  "detail-briefing": "围绕关注点展开细节，适合开题、复现或做文献综述前使用。",
  "paper-compare": "横向比较相关论文的问题、方法、实验线索、优点、局限和可复现切入点。",
  "paper-search": "只返回相关论文排序和证据片段，不生成长回答。",
  evaluation: "用内置评估集检查方向级检索是否能找回预期证据。",
  query: "直接向当前论文库提问，适合已有明确问题时使用。"
};

function App() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [mode, setMode] = useState<Mode>("direction-review");
  const [direction, setDirection] = useState("图神经网络在推荐系统中的应用");
  const [focus, setFocus] = useState("研究问题、代表方法、实验设置和可复现切入点");
  const [question, setQuestion] = useState("这个方向有哪些代表性方法？");
  const [docQuery, setDocQuery] = useState("");
  const [docKeyword, setDocKeyword] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [metadataSource, setMetadataSource] = useState("");
  const [doiFilter, setDoiFilter] = useState("");
  const [duplicateFilter, setDuplicateFilter] = useState("");
  const [documentSort, setDocumentSort] = useState("title");
  const [sectionFilter, setSectionFilter] = useState("");
  const [discoverySources, setDiscoverySources] = useState(["semantic_scholar", "openalex"]);
  const [topKDocuments, setTopKDocuments] = useState(3);
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

  function currentDocumentFilters(): DocumentFilters {
    return {
      query: docQuery,
      keyword: docKeyword,
      year_from: yearFrom,
      year_to: yearTo,
      source: metadataSource,
      has_doi: doiFilter,
      duplicate: duplicateFilter,
      sort_by: documentSort
    };
  }

  async function refreshDocuments(filters: DocumentFilters = currentDocumentFilters()) {
    const docs = await getDocuments(filters);
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

  async function handleApplyDocumentFilters() {
    setBusy(true);
    setError("");
    setStatus("正在筛选论文库");
    try {
      await refreshDocuments();
      setStatus("论文库筛选已应用");
    } catch (err) {
      setError(String(err));
      setStatus("筛选失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleClearDocumentFilters() {
    const emptyFilters = { sort_by: "title" };
    setDocQuery("");
    setDocKeyword("");
    setYearFrom("");
    setYearTo("");
    setMetadataSource("");
    setDoiFilter("");
    setDuplicateFilter("");
    setDocumentSort("title");
    setBusy(true);
    setError("");
    setStatus("正在清除论文库筛选");
    try {
      await refreshDocuments(emptyFilters);
      setStatus("论文库筛选已清除");
    } catch (err) {
      setError(String(err));
      setStatus("清除筛选失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleEnrichMetadata() {
    setBusy(true);
    setError("");
    setStatus("正在用 Crossref / Semantic Scholar 刷新元数据");
    try {
      const docs = await enrichMetadata();
      setDocuments(docs);
      setStatus("Crossref / Semantic Scholar 元数据已刷新");
    } catch (err) {
      setError(String(err));
      setStatus("元数据刷新失败");
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

  async function handleImportDiscoveredPaper(paper: DiscoveryPaper) {
    setBusy(true);
    setError("");
    setStatus(`导入 ${paper.title}`);
    try {
      await importDiscoveredPaper(paper);
      await refreshDocuments();
      if (result && "errors" in result) {
        setResult({
          ...result,
          papers: result.papers.map((item) =>
            item === paper ? { ...item, imported_document_id: "imported" } : item
          )
        });
      }
      setStatus("候选论文元数据已导入");
    } catch (err) {
      setError(String(err));
      setStatus("导入失败");
    } finally {
      setBusy(false);
    }
  }

  function toggleDiscoverySource(source: string) {
    setDiscoverySources((current) => {
      if (current.includes(source)) {
        const next = current.filter((item) => item !== source);
        return next.length ? next : current;
      }
      return [...current, source];
    });
  }

  function runCurrentTask() {
    if (mode === "discovery") {
      return searchDiscovery(direction, focus, discoverySources, topKDocuments);
    }
    if (mode === "query") {
      return askQuestion(question, Math.min(evidenceK, 10), sectionFilter);
    }
    if (mode === "paper-search") {
      return searchLiterature(direction, focus, topKDocuments, evidenceK, sectionFilter);
    }
    if (mode === "evaluation") {
      return runLiteratureEvaluation(topKDocuments, evidenceK, sectionFilter);
    }
    const task = mode === "direction-review" ? "review" : mode === "method-map" ? "methods" : mode === "paper-compare" ? "compare" : "details";
    return runLiteratureTask(task, direction, focus, topKDocuments, evidenceK, sectionFilter);
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
              <span className="subtle-count">{documents.length} 篇</span>
              <div className="panel-actions">
                <button className="icon-button" onClick={handleEnrichMetadata} disabled={busy} title="用 Crossref / Semantic Scholar 刷新元数据">
                  <FileSearch size={17} />
                </button>
                <button className="icon-button" onClick={handleReindex} disabled={busy} title="重新索引">
                  <RefreshCw size={17} />
                </button>
              </div>
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
            <div className="library-filters">
              <label className="field">
                <span>标题 / 作者 / DOI</span>
                <input value={docQuery} onChange={(event) => setDocQuery(event.target.value)} />
              </label>
              <label className="field">
                <span>关键词</span>
                <input value={docKeyword} onChange={(event) => setDocKeyword(event.target.value)} />
              </label>
              <div className="filter-grid">
                <label className="field">
                  <span>起始年份</span>
                  <input value={yearFrom} inputMode="numeric" onChange={(event) => setYearFrom(event.target.value)} />
                </label>
                <label className="field">
                  <span>结束年份</span>
                  <input value={yearTo} inputMode="numeric" onChange={(event) => setYearTo(event.target.value)} />
                </label>
                <label className="field">
                  <span>来源</span>
                  <select value={metadataSource} onChange={(event) => setMetadataSource(event.target.value)}>
                    <option value="">全部</option>
                    <option value="local">local</option>
                    <option value="crossref">crossref</option>
                    <option value="semantic_scholar">semantic_scholar</option>
                  </select>
                </label>
                <label className="field">
                  <span>DOI</span>
                  <select value={doiFilter} onChange={(event) => setDoiFilter(event.target.value)}>
                    <option value="">全部</option>
                    <option value="true">有 DOI</option>
                    <option value="false">无 DOI</option>
                  </select>
                </label>
                <label className="field">
                  <span>重复</span>
                  <select value={duplicateFilter} onChange={(event) => setDuplicateFilter(event.target.value)}>
                    <option value="">全部</option>
                    <option value="false">隐藏重复</option>
                    <option value="true">只看重复</option>
                  </select>
                </label>
                <label className="field">
                  <span>排序</span>
                  <select value={documentSort} onChange={(event) => setDocumentSort(event.target.value)}>
                    <option value="title">标题 A-Z</option>
                    <option value="year_desc">年份新到旧</option>
                    <option value="year_asc">年份旧到新</option>
                    <option value="citations_desc">引用数高到低</option>
                    <option value="references_desc">参考文献多到少</option>
                    <option value="source">来源</option>
                    <option value="filename">文件名</option>
                  </select>
                </label>
              </div>
              <div className="filter-actions">
                <button className="secondary-button" onClick={handleApplyDocumentFilters} disabled={busy}>
                  <Search size={15} />
                  应用筛选
                </button>
                <button className="secondary-button" onClick={handleClearDocumentFilters} disabled={busy}>
                  <RefreshCw size={15} />
                  清除
                </button>
              </div>
            </div>
            <div className="doc-list">
              {documents.map((doc) => (
                <article className="doc-row" key={doc.document_id}>
                  <FileText size={16} />
                  <div>
                    <strong>{displayTitle(doc)}</strong>
                    <span>{metadataLine(doc.metadata)} · {doc.pages} 页 · {doc.chunks} chunks</span>
                    <span>{sourceLine(doc.metadata)}</span>
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
                {item === "query" ? <MessageSquareText size={16} /> : item === "discovery" ? <Compass size={16} /> : <Network size={16} />}
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
            ) : mode === "evaluation" ? (
              <label className="field full">
                <span>评估说明</span>
                <input value="运行内置方向级检索评估集，检查预期关键词覆盖率。" readOnly />
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
            {mode === "discovery" ? (
              <div className="source-toggles">
                {[
                  ["semantic_scholar", "S2"],
                  ["crossref", "Crossref"],
                  ["arxiv", "arXiv"],
                  ["openalex", "OpenAlex"]
                ].map(([value, label]) => (
                  <label key={value}>
                    <input
                      type="checkbox"
                      checked={discoverySources.includes(value)}
                      onChange={() => toggleDiscoverySource(value)}
                    />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            ) : (
              <label className="field compact">
                <span>章节范围</span>
                <select value={sectionFilter} onChange={(event) => setSectionFilter(event.target.value)}>
                  {sectionOptions.map((option) => (
                    <option key={option.value || "all"} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="field compact">
              <span>{mode === "discovery" ? "每源数量" : "候选论文"}：{topKDocuments}</span>
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
              <span>{mode === "discovery" ? "外部发现" : "证据片段"}：{mode === "query" ? Math.min(evidenceK, 10) : evidenceK}</span>
              <input
                type="range"
                min={3}
                max={40}
                value={evidenceK}
                disabled={mode === "discovery"}
                onChange={(event) => setEvidenceK(Number(event.target.value))}
              />
            </label>
            <button className="primary-button" onClick={handleRun} disabled={busy}>
              {busy ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
              运行
            </button>
          </div>

          {error && <div className="error-box">{error}</div>}

          <ResultView result={result} onImportDiscoveredPaper={handleImportDiscoveredPaper} busy={busy} />
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

function ResultView({
  result,
  onImportDiscoveredPaper,
  busy
}: {
  result: AppResult;
  onImportDiscoveredPaper: (paper: DiscoveryPaper) => void;
  busy: boolean;
}) {
  if (!result) {
    return (
      <section className="result-empty">
        <Microscope size={32} />
        <p>输入研究方向后运行，系统会先找相关论文，再整理答案和证据。</p>
      </section>
    );
  }

  if ("cases" in result) {
    return <EvaluationView result={result} />;
  }

  if ("errors" in result) {
    return <DiscoveryView result={result} onImport={onImportDiscoveredPaper} busy={busy} />;
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

function DiscoveryView({
  result,
  onImport,
  busy
}: {
  result: DiscoveryResponse;
  onImport: (paper: DiscoveryPaper) => void;
  busy: boolean;
}) {
  return (
    <section className="result-stack">
      <article className="paper-panel">
        <div className="panel-heading">
          <h2>外部候选论文</h2>
          <span className="subtle-count">{result.papers.length} 篇 · {result.sources.join(" / ")}</span>
        </div>
        {result.errors.length > 0 && <div className="warning-box">{result.errors.join("；")}</div>}
        <div className="paper-grid discovery-grid">
          {result.papers.map((paper) => (
            <article className="paper-card discovery-card" key={`${paper.source}-${paper.source_id || paper.title}`}>
              <div className="paper-rank">{paper.source.slice(0, 2).toUpperCase()}</div>
              <div>
                <strong>{paper.title}</strong>
                <span>{discoveryLine(paper)}</span>
                {paper.doi && <span>DOI {paper.doi}</span>}
                {paper.fields_of_study.length > 0 && <span>{paper.fields_of_study.slice(0, 4).join(" · ")}</span>}
                <p>{paper.abstract || "暂无摘要。"}</p>
                <div className="paper-actions">
                  {paper.external_url && (
                    <a href={paper.external_url} target="_blank" rel="noreferrer">
                      查看来源
                    </a>
                  )}
                  {paper.pdf_url && (
                    <a href={paper.pdf_url} target="_blank" rel="noreferrer">
                      PDF
                    </a>
                  )}
                  <button
                    className="secondary-button"
                    onClick={() => onImport(paper)}
                    disabled={busy || Boolean(paper.imported_document_id)}
                  >
                    <Plus size={14} />
                    {paper.imported_document_id ? "已导入" : "导入元数据"}
                  </button>
                </div>
              </div>
            </article>
          ))}
        </div>
      </article>
    </section>
  );
}

function EvaluationView({ result }: { result: LiteratureEvaluationResponse }) {
  return (
    <section className="result-stack">
      <article className="answer-panel">
        <div className="answer-meta">
          <span>{result.retrieval_mode}</span>
          <span>{result.passed_cases}/{result.total_cases} passed</span>
          <span>avg {result.average_score.toFixed(2)}</span>
        </div>
        <div className="evaluation-grid">
          {result.cases.map((item) => (
            <article className="evaluation-card" key={item.name}>
              <div className="evaluation-head">
                <strong>{item.name}</strong>
                <span className={item.passed ? "pass" : "warn"}>{item.score.toFixed(2)}</span>
              </div>
              <p>{item.query}</p>
              <small>{item.focus || "no focus"} · {item.section_filter || "all sections"}</small>
              <span>命中：{item.matched_terms.length ? item.matched_terms.join(", ") : "none"}</span>
              <span>缺失：{item.missing_terms.length ? item.missing_terms.join(", ") : "none"}</span>
              <span>论文：{item.papers.length} · 证据：{item.sources.length}</span>
            </article>
          ))}
        </div>
      </article>
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
              {paper.evidence_sections.length > 0 && (
                <span>章节 {paper.evidence_sections.map(sectionLabel).join(" / ")}</span>
              )}
              {paper.metadata.doi && <span>DOI {paper.metadata.doi}</span>}
              <span>{sourceLine(paper.metadata)}</span>
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
          [{index}] {source.filename} · 第 {source.page} 页 · {sectionLabel(source.section)}
        </strong>
        <span>{source.score.toFixed(3)}</span>
      </div>
      <p>{source.text}</p>
      <small>{source.chunk_id}</small>
    </article>
  );
}

function sectionLabel(section: string) {
  return sectionLabels[section] ?? section;
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

function sourceLine(metadata: {
  metadata_source: string;
  publisher: string | null;
  reference_count: number | null;
  citation_count: number | null;
  metadata_confidence: string;
  metadata_match_score: number | null;
  external_url: string | null;
  is_enriched: boolean;
}) {
  const parts = [
    metadata.is_enriched ? `source ${metadata.metadata_source}` : "source local",
    metadata.metadata_confidence !== "local" ? `confidence ${metadata.metadata_confidence}` : null,
    metadata.publisher,
    metadata.reference_count !== null ? `${metadata.reference_count} refs` : null,
    metadata.citation_count !== null ? `${metadata.citation_count} citations` : null,
    metadata.metadata_match_score !== null ? `match ${metadata.metadata_match_score.toFixed(2)}` : null,
    metadata.external_url
  ].filter(Boolean);
  return parts.join(" · ");
}

function discoveryLine(paper: DiscoveryPaper) {
  const parts = [
    paper.year ? String(paper.year) : null,
    paper.venue,
    paper.authors,
    `${paper.relevance_score.toFixed(2)} relevance`,
    paper.citation_count !== null ? `${paper.citation_count} citations` : null,
    paper.is_open_access ? "open access" : null
  ].filter(Boolean);
  return parts.join(" · ");
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
