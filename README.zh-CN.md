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
10. 对同一研究方向下的代表论文做横向对比。

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

### V11：章节感知检索

V11 增加了章节感知检索。

每个 chunk 现在都会保存一个粗粒度论文节标签，例如：

```text
abstract
introduction
related_work
methods
experiments
results
discussion
conclusion
references
unknown
```

`/query`、`/study/*` 和 `/literature/*` 都支持可选字段 `section_filter`。这样你可以只检索论文的某个部分，例如只看 `methods`，或者只看 `experiments`。

来源片段现在会返回：

```text
section
```

相关论文候选还会返回：

```text
evidence_sections
```

前端新增了章节筛选控件，并且会在每个证据卡片上显示章节标签。

### V12：论文库筛选与管理

V12 增加了本地论文库筛选和排序。

`GET /documents` 现在支持这些查询参数：

```text
query
keyword
year_from
year_to
source
has_doi
duplicate
sort_by
```

前端论文库现在支持：

```text
按标题、作者、DOI、摘要搜索
按关键词搜索
按年份范围筛选
按元数据来源筛选：local、crossref、semantic_scholar
按是否有 DOI 筛选
按是否疑似重复筛选
按标题、年份、引用数、参考文献数量、来源、文件名排序
```

V12 是纯本地功能，不接 Elsevier 或其他期刊 API。它的目标是先让已有后台论文库更容易管理，等本地管理稳定后，再接外部数据源。

### V13：方向级检索质量评估

V13 增加了一个轻量的方向级检索评估工作流。

后端现在内置了一个小型评估集。每个评估 case 会运行现有的文献检索流程，然后检查返回的论文和证据片段是否覆盖预期关键词，并计算一个简单分数。

V13 新增接口：

```text
POST /evaluation/literature
```

前端新增 `检索评估` 模式，会展示：

```text
通过的 case 数量
平均分
命中的关键词
缺失的关键词
返回论文数量
返回证据数量
```

这还不是完整学术 benchmark，而是一个本地回归检查工具。它的作用是帮助我们判断后续修改检索逻辑后，方向级检索效果是变好了还是变差了。

### V14：论文对比工作流

V14 增加了方向级论文对比接口。

新接口：

```text
POST /literature/compare
```

用户输入研究方向和关注重点后，系统会先检索代表论文、聚合证据片段、按论文排序，然后基于证据对比：

```text
研究问题
方法差异
实验和数据线索
优点
局限
适合继续研究或复现的切入点
```

如果没有配置 LLM，这个接口仍然可以以 retrieval-only 模式返回相关论文排序和证据片段。

### V15：外部文献发现

V15 增加了外部文献发现能力。

系统不再只依赖已经上传到本地论文库的 PDF，也可以根据研究方向去公开文献元数据源搜索候选论文：

```text
Semantic Scholar
Crossref
arXiv
OpenAlex
```

V15 新增接口：

```text
POST /discovery/search
POST /discovery/import-metadata
```

`/discovery/search` 会返回外部候选论文，包括标题、作者、年份、venue、DOI、摘要、引用数、参考文献数、来源链接、可用 PDF 链接、是否 open access，以及本地是否已经导入。

文献发现层会根据用户输入的研究方向和关注重点做关键词相关性过滤。候选论文需要在标题、摘要、关键词、研究领域、venue、DOI 或相关元数据中覆盖输入关键词，才会展示出来。

`/discovery/import-metadata` 会把选中的候选论文先作为“仅元数据记录”导入本地论文库。它不会自动下载 PDF；导入后的记录 `pages=0`，但会生成一个可检索的元数据 chunk，所以在上传完整 PDF 之前，也可以参与本地检索、方向综述、方法梳理和论文对比。

前端现在也会展示当前模式的提示词或检索意图，方便用户检查每个工作流到底在要求系统做什么。

### V16：工程化启动和质量门禁

V16 增加了本地工程化能力：

```text
scripts/check-env.ps1
scripts/start.ps1
scripts/stop.ps1
ci/github-actions-ci.yml
CHANGELOG.md
```

现在可以一键检查环境、启动前后端、停止本地服务，并提供 GitHub Actions CI 模板用于后端测试和前端构建。版本号更新为 `0.16.0`。

如果需要启用 GitHub Actions，把 `ci/github-actions-ci.yml` 复制到 `.github/workflows/ci.yml`，并用具有 `workflow` 权限的 GitHub 凭据推送。

## 项目结构

```text
backend/
  app/
    discovery.py
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

## 推荐启动方式

先做环境自检：

```powershell
.\scripts\check-env.ps1
```

一键启动前后端：

```powershell
.\scripts\start.ps1
```

启动后会打开：

```text
http://127.0.0.1:5173
```

停止本地前后端端口：

```powershell
.\scripts\stop.ps1
```

如果喜欢双击运行，也可以使用：

```text
scripts\check-env.cmd
scripts\start.cmd
scripts\stop.cmd
```

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

V12 后该接口支持可选筛选：

```text
GET /documents?query=water&keyword=remote&year_from=2024&source=crossref&has_doi=true&duplicate=false&sort_by=year_desc
```

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
  "top_k": 4,
  "section_filter": "methods"
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
  "top_k": 6,
  "section_filter": "experiments"
}
```

### 方向级文献工作流

```text
POST /literature/search
POST /literature/review
POST /literature/methods
POST /literature/details
POST /literature/compare
```

示例：

```json
{
  "query": "water quality prediction neural networks",
  "focus": "methods and experiments",
  "top_k_documents": 3,
  "evidence_k": 8,
  "section_filter": "methods"
}
```

用途：

```text
/literature/search   只返回相关论文排序和证据片段
/literature/review   生成方向级文献综述
/literature/methods  梳理方法类别、技术细节、优缺点和对应论文
/literature/details  围绕关注点做细节分析，适合开题、复现或深入阅读
/literature/compare  横向对比代表论文的问题、方法、实验、优点、局限和后续切入点
```

### 外部文献发现

```text
POST /discovery/search
POST /discovery/import-metadata
```

搜索示例：

```json
{
  "query": "graph neural networks for recommendation",
  "focus": "survey and benchmark papers",
  "sources": ["semantic_scholar", "crossref", "arxiv", "openalex"],
  "limit_per_source": 5
}
```

用 `/discovery/search` 搜索本地论文库之外的候选论文。用 `/discovery/import-metadata` 将选中的候选论文先保存到本地论文库，之后再上传 PDF 或补全文本。

### 方向级检索评估

```text
POST /evaluation/literature
```

示例：

```json
{
  "top_k_documents": 5,
  "evidence_k": 18,
  "section_filter": null
}
```

该接口会运行内置评估 case，并返回每个 case 的命中词、缺失词、得分、相关论文和证据片段。

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

## V11 工作流程

```text
PDF 上传或重新索引
  -> 提取页面文本
  -> 将文本切分成 chunks
  -> 根据附近章节标题推断粗粒度 section
  -> 将 section 保存到 data/index/rag_store.json
  -> 检索时根据可选 section_filter 限定证据范围
  -> API 返回带 section 的来源片段和论文候选
  -> 前端可以只看摘要、方法、实验、结果或结论
```

## V12 工作流程

```text
论文库筛选条件
  -> 前端收集本地筛选参数
  -> GET /documents 接收查询参数
  -> 后端筛选已有本地论文元数据
  -> 后端对筛选后的论文排序
  -> 前端展示更小、更容易管理的论文列表
```

## V13 工作流程

```text
内置评估 case
  -> 对每个 case 运行文献检索
  -> 收集返回的论文和证据片段
  -> 检查证据是否覆盖预期关键词
  -> 计算每个 case 的得分和是否通过
  -> 计算所有 case 的平均分
  -> 前端展示检索评估报告
```

## V14 工作流程

```text
研究方向
  -> 从论文库中检索候选证据片段
  -> 按论文聚合候选结果
  -> 排序得到代表论文
  -> 基于证据构造论文对比 prompt
  -> 可选 LLM 生成论文对比
  -> 前端展示对比结果、论文排序和证据片段
```

## 下一步计划

1. 为 open-access 候选论文增加安全 PDF 下载。
2. 增加 BibTeX / Markdown 导出。
3. 增加论文状态标签，例如待读、阅读中、已读、忽略、重点。
4. 等 API key 和访问权限具备后，接入 Elsevier、Scopus、ScienceDirect。
