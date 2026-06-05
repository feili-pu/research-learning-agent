# Research Learning Agent

Research Learning Agent 是一个面向研究生文献学习和方向调研的本地 RAG 项目。它的目标不是让用户逐篇读完所有论文，而是维护一个后台论文库，然后根据用户输入的研究方向，检索相关论文、梳理方法体系、提炼细节和证据来源。

## 当前能力

当前版本已经支持：

1. 上传 PDF 到本地论文库。
2. 从 PDF 中抽取文本。
3. 将论文文本切分成可检索的 chunks。
4. 使用语义向量进行检索，并保留 TF-IDF fallback。
5. 调用 OpenAI 兼容接口生成带来源依据的回答。
6. 提供方向综述、方法梳理、细节分析和相关论文排序。
7. 自动提取论文元数据，例如标题、作者、年份、venue、DOI、摘要和关键词。
8. 识别疑似重复论文，例如相同 DOI、相同标题或相同文件名。
9. 提供一个科学风格的 React 前端工作台。

## 版本演进

### V1：最小 PDF 问答

V1 实现了最基础的 PDF RAG 流程：

```text
上传 PDF -> 提取文本 -> 切分 chunks -> TF-IDF 检索 -> 返回相关片段
```

这一版重点是把完整链路跑通。

### V2：语义检索

V2 默认使用 `sentence-transformers` 做语义向量检索。

默认 embedding 模型是：

```text
BAAI/bge-small-zh-v1.5
```

如果语义模型不可用，系统会自动 fallback 到 TF-IDF。

### V3：LLM 回答生成

V3 增加了可选 LLM 生成能力。

如果配置了 `OPENAI_API_KEY`，系统会把检索到的来源片段交给 LLM，并要求回答中使用 `[1]`、`[2]` 这样的引用编号。

如果没有配置 API key，系统仍然可以本地运行，只返回 retrieval-only 答案和来源片段。

### V4：本地索引持久化

V4 将上传后的论文信息和 chunks 保存到：

```text
data/index/rag_store.json
```

这样 API 重启后不会忘记之前上传过的文档。

### V5：学习工作流

V5 增加了结构化学习接口：

```text
POST /study/summary
POST /study/key-points
POST /study/reading-plan
```

这几个接口适合对当前资料做总览、关键点提取和阅读计划。

### V6：前端工作台

V6 增加了 React + Vite 前端。

前端支持：

```text
PDF 上传
论文库列表
重新索引
自由问答
资料总览
关键点提取
阅读计划
来源片段展示
检索模式和回答模式展示
```

前端地址：

```text
http://127.0.0.1:5173
```

后端地址：

```text
http://127.0.0.1:8000
```

### V7：方向级文献研究

V7 将产品定位从“单篇或少量论文学习工具”升级为“方向级文献研究工作台”。

用户可以输入一个研究方向，例如：

```text
water quality prediction neural networks
graph neural networks for recommendation
remote sensing change detection
```

系统会：

1. 在整个后台论文库中检索相关证据片段。
2. 按 `document_id` 聚合为论文级候选。
3. 对相关论文进行排序。
4. 基于证据生成方向综述、方法梳理或细节分析。

V7 新增接口：

```text
POST /literature/search
POST /literature/review
POST /literature/methods
POST /literature/details
```

### V8：论文元数据和去重

V8 将后台 PDF 集合升级成更像真实论文库的结构。

上传或重新索引 PDF 时，系统会从论文首页和前几页中本地启发式提取：

```text
title
authors
year
venue
doi
abstract
keywords
duplicate_of
duplicate_reason
```

当前重复检测规则：

```text
same DOI
same normalized title
same filename
```

前端现在会在论文库和相关论文排序结果中展示论文标题、年份、DOI、摘要预览和疑似重复标记。

注意：V8 暂时不调用外部元数据服务。元数据提取是本地启发式规则，所以少数 PDF 的作者、摘要或关键词可能还不完美。后续可以接入 Crossref、Semantic Scholar 或 arXiv 做增强。

### V9：Crossref DOI 元数据增强

V9 增加了 Crossref DOI 元数据增强。

如果本地 PDF 解析已经提取到了 DOI，后端可以调用 Crossref Works API，用 DOI 查询官方登记信息，并补全或修正：

```text
标题
作者
年份
期刊/会议
出版社
DOI
官方链接
参考文献数量
学科/关键词
Crossref 提供的摘要
```

V9 新增接口：

```text
POST /documents/enrich-metadata
```

前端论文库中新增了一个 Crossref 刷新按钮。元数据现在还会包含：

```text
metadata_source: local 或 crossref
is_enriched: true 或 false
```

### V10：Semantic Scholar 标题检索 fallback

V10 增加了 Semantic Scholar 标题检索 fallback。

元数据增强接口仍然会优先使用 Crossref：如果论文有 DOI，就先用 DOI 去查 Crossref。若论文没有 DOI，或者 Crossref 没有返回可用记录，后端会用论文标题去 Semantic Scholar 搜索，并且只接受标题相似度足够高的结果。

V10 可以为没有 DOI 的论文补全：

```text
标题
作者
年份
期刊/会议
Semantic Scholar 暴露的 DOI
摘要
官方链接
参考文献数量
引用数量
研究领域
元数据置信度
标题匹配分数
```

元数据现在支持：

```text
metadata_source: local、crossref 或 semantic_scholar
metadata_confidence: local、low、medium 或 high
metadata_match_score: Semantic Scholar 标题匹配分数
```

## 项目结构

```text
backend/
  app/
    main.py
    rag.py
    schemas.py
    literature.py
    study.py
    answerer.py
data/
  uploads/
frontend/
  src/
tests/
```

## 环境配置

本机当前使用的 conda 环境是：

```text
graph-rag
```

安装或刷新 Python 依赖：

```powershell
conda run -n graph-rag python -m pip install -r requirements.txt
```

可选 LLM 配置：

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:RLA_OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:RLA_LLM_MODEL="gpt-4o-mini"
$env:RLA_OPENAI_WIRE_API="responses"
```

也可以把这些配置写入本地 `.env` 文件。`.env` 已经被 Git 忽略，不应该提交真实 API key。

## 启动后端

```powershell
conda run -n graph-rag uvicorn backend.app.main:app --reload
```

打开 Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

## 启动前端

第一次运行前端需要安装依赖：

```powershell
cd frontend
npm install
```

启动 Vite 开发服务器：

```powershell
npm run dev -- --host 127.0.0.1 --port 5173
```

然后打开：

```text
http://127.0.0.1:5173
```

前端通过 `/api` 代理访问后端，所以需要保持 FastAPI 后端运行在 `8000` 端口。

## API 接口

### 健康检查

```text
GET /health
```

### 上传 PDF

```text
POST /documents/upload
```

可以在 Swagger UI 中通过表单上传 PDF。

### 查看论文库

```text
GET /documents
```

V8 后该接口会返回论文元数据。

### 重新索引论文库

```text
POST /documents/reindex
```

该接口会扫描 `data/uploads/` 中已有 PDF，重新抽取文本、重建 chunks、刷新语义索引，并重新提取元数据和重复标记。

### 用 Crossref 刷新元数据

```text
POST /documents/enrich-metadata
```

该接口会对已经有 DOI 的论文调用 Crossref，尝试用官方元数据补全论文信息。

如果 Crossref 无法补全某篇论文，V10 会继续用 Semantic Scholar 标题搜索作为 fallback。

### 自由问答

```text
POST /query
```

示例：

```json
{
  "question": "What is retrieval augmented generation?",
  "top_k": 4
}
```

### 学习工作流

```text
POST /study/summary
POST /study/key-points
POST /study/reading-plan
```

示例：

```json
{
  "topic": "这些文档",
  "focus": "研究方法和实验结论",
  "top_k": 6
}
```

### 方向级文献工作流

```text
POST /literature/search
POST /literature/review
POST /literature/methods
POST /literature/details
```

示例：

```json
{
  "query": "water quality prediction neural networks",
  "focus": "methods and experiments",
  "top_k_documents": 3,
  "evidence_k": 8
}
```

用途：

```text
/literature/search   只返回相关论文排序和证据片段
/literature/review   生成方向级文献综述
/literature/methods  梳理方法类别、技术细节、优缺点和对应论文
/literature/details  围绕关注点做细节分析，适合开题、复现或深入阅读
```

## V8 工作流程

```text
PDF 上传或重新索引
  -> 提取首页和前几页文本
  -> 推断标题、作者、年份、venue、DOI、摘要和关键词
  -> 根据 DOI、标题或文件名标记疑似重复论文
  -> 保存到 data/index/rag_store.json
  -> API 返回带 metadata 的论文对象
  -> 前端以论文卡片形式展示，而不是只显示 PDF 文件名
```

## V9 工作流程

```text
已有 DOI 的论文元数据
  -> 通过 DOI 调用 Crossref Works API
  -> 映射官方标题、作者、年份、venue、publisher、URL 和 reference count
  -> 将 metadata_source 标记为 crossref
  -> 保存增强后的元数据到 data/index/rag_store.json
  -> 前端展示增强来源和更准确的文献信息
```

## V10 工作流程

```text
已有论文元数据
  -> 有 DOI 时优先调用 Crossref
  -> Crossref 没有可用结果时，用标题搜索 Semantic Scholar
  -> 比较 Semantic Scholar 返回标题和本地标题的相似度
  -> 只接受超过阈值的匹配结果
  -> 保存来源、置信度、引用数量、研究领域和匹配分数
  -> 前端展示增强来源和置信度信息
```

## 下一步计划

1. 增加章节感知检索，区分 Abstract、Methods、Results、Conclusion。
2. 给论文库增加年份、DOI、重复状态、来源、关键词筛选。
3. 增加方向级检索质量评估数据。
4. 增加 Agent 工具，让系统自动查 arXiv、Semantic Scholar 和 GitHub。
5. 增加论文对比工作流。
