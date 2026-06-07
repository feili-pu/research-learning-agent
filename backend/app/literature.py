from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
import json
import os
import re

from .answerer import Answerer
from .presenters import paper_metadata, source_chunks
from .rag import Document, RagStore, SearchResult
from .schemas import (
    LiteratureRequest,
    LiteratureReviewResponse,
    LiteratureRetrievalTrace,
    LiteratureSearchResponse,
    PaperCandidate,
)


@dataclass(frozen=True)
class QueryIntent:
    search_query: str
    required_groups: list[list[str]] = field(default_factory=list)
    relevance_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    query_rewrites: list[str] = field(default_factory=list)
    planner: str = "rules"
    planner_model: str | None = None
    planner_error: str | None = None


@dataclass(frozen=True)
class DocumentCandidate:
    document: Document
    results: list[SearchResult]
    score: float
    topic_score: int
    rerank_reason: str | None = None


class LiteratureService:
    def __init__(self, store: RagStore, answerer: Answerer) -> None:
        self.store = store
        self.answerer = answerer

    def search(self, request: LiteratureRequest) -> LiteratureSearchResponse:
        intent, papers, filtered_results, trace = self._retrieve(request)
        return LiteratureSearchResponse(
            query=request.query,
            retrieval_mode=self.store.active_retrieval_mode,
            retrieval_trace=trace,
            papers=papers,
            sources=source_chunks(filtered_results),
        )

    def review(self, request: LiteratureRequest) -> LiteratureReviewResponse:
        prompt = self._build_prompt(
            task="direction_review",
            request=request,
            instruction=(
                "请把这些论文当作一个后台论文库来使用，围绕用户给定研究方向生成中文文献综述。"
                "输出应包含：研究背景、核心问题、代表论文、方法分类、主要结论、局限性、可继续深入的问题。"
                "不要试图总结整个论文库，只总结与研究方向相关的论文。"
                "禁止把其他方向或其他应用领域的方法迁移、类比或改写成当前方向的已有文献。"
            ),
        )
        return self._run("direction_review", request, prompt)

    def methods(self, request: LiteratureRequest) -> LiteratureReviewResponse:
        prompt = self._build_prompt(
            task="method_map",
            request=request,
            instruction=(
                "请专注梳理这个研究方向中的方法体系。"
                "输出应包含：方法类别、每类方法解决的问题、关键技术细节、优点、局限、对应论文来源。"
                "如果证据不足，请明确指出缺少哪些方法细节。"
                "只总结已有文献中实际属于该方向的方法，禁止引入其他领域方法迁移建议。"
            ),
        )
        return self._run("method_map", request, prompt)

    def details(self, request: LiteratureRequest) -> LiteratureReviewResponse:
        prompt = self._build_prompt(
            task="detail_briefing",
            request=request,
            instruction=(
                "请围绕用户关注点做细节梳理，适合研究生做开题或复现前阅读。"
                "输出应包含：关键定义、实验/数据线索、模型或算法细节、重要结论、可复现切入点。"
                "只依据与研究方向直接相关的已有文献，不要用其他领域论文补充迁移思路。"
            ),
        )
        return self._run("detail_briefing", request, prompt)

    def compare(self, request: LiteratureRequest) -> LiteratureReviewResponse:
        prompt = self._build_prompt(
            task="paper_compare",
            request=request,
            instruction=(
                "请对检索到的代表论文做横向对比，帮助用户判断哪些论文值得优先阅读或复现。"
                "输出应包含：对比维度表、每篇论文的核心问题、方法差异、实验/数据线索、优点、局限、适合继续研究的切入点。"
                "如果某个维度证据不足，请写明缺少证据，不要补编。"
                "只比较与研究方向直接相关的已有文献，不要把其他领域论文当作可迁移方法纳入对比。"
            ),
        )
        return self._run("paper_compare", request, prompt)

    def _run(
        self,
        task: str,
        request: LiteratureRequest,
        prompt: str,
    ) -> LiteratureReviewResponse:
        _, papers, filtered_results, trace = self._retrieve(request)
        if not papers:
            return LiteratureReviewResponse(
                task=task,
                query=request.query,
                retrieval_mode=self.store.active_retrieval_mode,
                answer_mode="no_relevant_papers",
                model=None,
                answer=(
                    "当前本地论文库中没有找到与该研究方向直接相关的已有文献。"
                    "我没有使用其他领域论文做方法迁移或类比总结。"
                    "请先通过“文献发现”导入该方向候选论文，或上传相关 PDF 后再运行。"
                ),
                retrieval_trace=trace,
                papers=[],
                sources=[],
            )
        if self._requires_evidence_guard(trace) and trace.evidence_coverage < 0.6:
            return LiteratureReviewResponse(
                task=task,
                query=request.query,
                retrieval_mode=self.store.active_retrieval_mode,
                answer_mode="insufficient_evidence",
                model=None,
                answer=(
                    "已找到一些候选论文，但证据没有覆盖该研究方向的核心主题。"
                    f"当前证据覆盖率为 {trace.evidence_coverage:.2f}。"
                    "为避免把其他领域论文迁移成该方向已有文献，我没有生成综述。"
                    "建议先导入更多直接相关论文或放宽/修正研究方向后再试。"
                ),
                retrieval_trace=trace,
                papers=papers,
                sources=source_chunks(filtered_results),
            )
        answer = self.answerer.answer(prompt, filtered_results)
        return LiteratureReviewResponse(
            task=task,
            query=request.query,
            retrieval_mode=self.store.active_retrieval_mode,
            answer_mode=answer.answer_mode,
            model=answer.model,
            answer=answer.answer,
            retrieval_trace=trace,
            papers=papers,
            sources=source_chunks(filtered_results),
        )

    def _retrieve(
        self,
        request: LiteratureRequest,
    ) -> tuple[QueryIntent, list[PaperCandidate], list[SearchResult], LiteratureRetrievalTrace]:
        intent = self._query_intent(request)
        chunk_results = self.store.search(
            intent.search_query,
            self._candidate_evidence_k(request),
            request.section_filter,
            allow_semantic=False,
        )
        recalled_candidates = self._recall_document_candidates(chunk_results, intent, request)
        gated_candidates = [
            candidate
            for candidate in recalled_candidates
            if self._document_matches_topic(candidate.document.document_id, candidate.results, intent)
        ]
        candidates = self._dedupe_document_candidates(gated_candidates)
        candidates, reranker, reranker_error = self._rerank_document_candidates(candidates, intent, request)
        candidates = candidates[: request.top_k_documents]
        evidence_by_document = self._extract_evidence(candidates, intent, request)
        papers = [
            self._paper_candidate(candidate, evidence_by_document.get(candidate.document.document_id, []))
            for candidate in candidates
        ]
        filtered_results = []
        for paper in papers:
            filtered_results.extend(evidence_by_document.get(paper.document_id, []))
        excluded_titles = self._excluded_candidate_titles(recalled_candidates, gated_candidates)
        evidence_coverage = self._evidence_coverage(intent, filtered_results)
        trace = LiteratureRetrievalTrace(
            query_planner=intent.planner,
            planner_model=intent.planner_model,
            planner_error=intent.planner_error,
            search_query=intent.search_query,
            query_rewrites=intent.query_rewrites,
            required_groups=intent.required_groups,
            relevance_terms=intent.relevance_terms,
            exclude_terms=intent.exclude_terms,
            reranker=reranker,
            reranker_model=getattr(self.answerer, "model", None) if reranker.startswith("llm") else None,
            reranker_error=reranker_error,
            evidence_coverage=evidence_coverage,
            candidate_count=len(recalled_candidates),
            gated_count=len(gated_candidates),
            returned_count=len(papers),
            excluded_titles=excluded_titles[:8],
        )
        return intent, papers, filtered_results[: request.evidence_k], trace

    def _candidate_evidence_k(self, request: LiteratureRequest) -> int:
        return min(max(request.evidence_k * 4, request.evidence_k), 80)

    def _search_query(self, request: LiteratureRequest) -> str:
        if not request.focus:
            return request.query
        return f"{request.query}\n关注重点：{request.focus}"

    def _query_intent(self, request: LiteratureRequest) -> QueryIntent:
        rule_intent = self._rule_query_intent(request)
        llm_intent = self._llm_query_intent(request, rule_intent)
        if llm_intent is None:
            return rule_intent
        return llm_intent

    def _rule_query_intent(self, request: LiteratureRequest) -> QueryIntent:
        raw_query = self._search_query(request)
        required_groups: list[list[str]] = []
        relevance_terms = self._topic_tokens(raw_query)

        expansion_map = {
            "桑树": ["桑树", "mulberry"],
            "桑叶": ["桑叶", "mulberry", "leaf"],
            "病虫害": ["病虫害", "disease", "pest"],
            "病害": ["病害", "disease"],
            "虫害": ["虫害", "pest"],
            "叶片": ["叶片", "leaf"],
            "检测": ["检测", "detection"],
            "识别": ["识别", "recognition", "detection"],
            "分类": ["分类", "classification"],
            "图神经网络": ["图神经网络", "graph neural network", "gnn"],
            "推荐系统": ["推荐系统", "recommender", "recommendation"],
            "水质": ["水质", "water quality"],
            "遥感": ["遥感", "remote sensing"],
            "高光谱": ["高光谱", "hyperspectral"],
            "无人机": ["无人机", "uav", "unmanned aerial"],
            "生物识别": ["生物识别", "biometric"],
            "模板保护": ["模板保护", "template protection"],
        }

        for phrase, expansions in expansion_map.items():
            if "病虫害" in raw_query and phrase in {"病害", "虫害"}:
                continue
            if phrase in raw_query:
                group = self._unique_terms(expansions)
                required_groups.append(group)
                relevance_terms.extend(group)

        english_groups = [
            (("mulberry",), ["mulberry"]),
            (("leaf", "disease"), ["leaf disease", "disease"]),
            (("pest",), ["pest"]),
            (("graph", "neural"), ["graph neural network", "gnn"]),
            (("recommend",), ["recommender", "recommendation"]),
            (("water", "quality"), ["water quality"]),
            (("biometric",), ["biometric"]),
        ]
        normalized_raw = self._normalize_topic_text(raw_query)
        for needles, expansions in english_groups:
            if all(needle in normalized_raw for needle in needles):
                group = self._unique_terms(expansions)
                required_groups.append(group)
                relevance_terms.extend(group)

        relevance_terms = self._unique_terms(relevance_terms)
        search_terms = relevance_terms + self._topic_tokens(request.focus or "")
        search_query = " ".join(self._unique_terms(search_terms)) or raw_query
        return QueryIntent(
            search_query=search_query,
            required_groups=self._dedupe_groups(required_groups),
            relevance_terms=relevance_terms,
            query_rewrites=[search_query],
            planner="rules",
        )

    def _llm_query_intent(
        self,
        request: LiteratureRequest,
        fallback: QueryIntent,
    ) -> QueryIntent | None:
        if getattr(self.answerer, "client", None) is None:
            return None

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._call_query_intent_llm, request, fallback)
            payload = future.result(timeout=self._query_planner_timeout())
        except FuturesTimeoutError:
            return QueryIntent(
                search_query=fallback.search_query,
                required_groups=fallback.required_groups,
                relevance_terms=fallback.relevance_terms,
                exclude_terms=fallback.exclude_terms,
                query_rewrites=fallback.query_rewrites,
                planner="rules",
                planner_model=getattr(self.answerer, "model", None),
                planner_error="LLM query parsing timed out; fell back to rules.",
            )
        except Exception as exc:
            return QueryIntent(
                search_query=fallback.search_query,
                required_groups=fallback.required_groups,
                relevance_terms=fallback.relevance_terms,
                exclude_terms=fallback.exclude_terms,
                query_rewrites=fallback.query_rewrites,
                planner="rules",
                planner_model=getattr(self.answerer, "model", None),
                planner_error=f"LLM query parsing failed: {type(exc).__name__}",
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        query_rewrites = self._unique_terms(
            [str(item) for item in payload.get("query_rewrites", [])]
        )[:4]
        relevance_terms = self._unique_terms(
            [
                *fallback.relevance_terms,
                *[str(item) for item in payload.get("core_terms", [])],
                *[str(item) for item in payload.get("task_terms", [])],
                *self._topic_tokens(" ".join(query_rewrites)),
            ]
        )[:24]
        required_groups = [
            *fallback.required_groups,
            *self._normalize_groups(payload.get("required_groups", [])),
        ]
        exclude_terms = self._unique_terms(
            [str(item) for item in payload.get("exclude_terms", [])]
        )[:16]
        search_query = " ".join(self._unique_terms([*query_rewrites, *relevance_terms])) or fallback.search_query
        return QueryIntent(
            search_query=search_query,
            required_groups=self._dedupe_groups(required_groups),
            relevance_terms=relevance_terms or fallback.relevance_terms,
            exclude_terms=exclude_terms,
            query_rewrites=query_rewrites or fallback.query_rewrites,
            planner="llm",
            planner_model=getattr(self.answerer, "model", None),
        )

    def _call_query_intent_llm(
        self,
        request: LiteratureRequest,
        fallback: QueryIntent,
    ) -> dict:
        prompt = (
            "你是文献检索查询解析器。请把用户研究方向解析成严格 JSON，不要输出 Markdown。\n"
            "目标是只检索已有文献，不做跨领域方法迁移。\n"
            "JSON 字段：\n"
            "- query_rewrites: 2-4 个中英混合检索式，优先英文论文检索表达\n"
            "- core_terms: 6-12 个核心主题词或同义词\n"
            "- task_terms: 2-8 个任务/方法词\n"
            "- required_groups: 数组的数组；每个子数组是一组同义必需主题，论文至少要命中每组之一\n"
            "- exclude_terms: 与该方向容易混淆但应排除的主题词\n\n"
            f"研究方向：{request.query}\n"
            f"关注重点：{request.focus or ''}\n"
            f"规则解析检索式：{fallback.search_query}\n"
            f"规则必需组：{fallback.required_groups}\n"
        )
        output = self.answerer.complete(
            prompt,
            system=(
                "You are a strict literature retrieval query parser. "
                "Return one valid JSON object only. Do not answer the research question."
            ),
        )
        return self._parse_json_object(output)

    def _query_planner_timeout(self) -> float:
        return float(os.getenv("RLA_QUERY_PLANNER_TIMEOUT", "45"))

    def _reranker_timeout(self) -> float:
        return float(os.getenv("RLA_RERANKER_TIMEOUT", "45"))

    def _parse_json_object(self, value: str) -> dict:
        text = value.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}

    def _normalize_groups(self, value) -> list[list[str]]:
        groups: list[list[str]] = []
        if not isinstance(value, list):
            return groups
        for item in value:
            if isinstance(item, list):
                group = self._unique_terms([str(term) for term in item])
            else:
                group = self._unique_terms([str(item)])
            if group:
                groups.append(group)
        return groups

    def _dedupe_groups(self, groups: list[list[str]]) -> list[list[str]]:
        deduped = []
        seen = set()
        for group in groups:
            key = tuple(group)
            if group and key not in seen:
                deduped.append(group)
                seen.add(key)
        return deduped

    def _build_prompt(
        self,
        task: str,
        request: LiteratureRequest,
        instruction: str,
    ) -> str:
        focus = f"\n关注重点：{request.focus}" if request.focus else ""
        section = f"\n章节范围：{request.section_filter}" if request.section_filter else ""
        return (
            f"任务：{task}\n"
            f"研究方向：{request.query}"
            f"{focus}\n"
            f"{section}"
            f"要求：{instruction}\n"
            "回答时必须基于给定来源，并用 [1]、[2] 这样的编号引用证据。"
            "如果来源不足，只能说明本地已有文献不足，不能提出跨领域迁移方案来填补。"
        )

    def _recall_document_candidates(
        self,
        chunk_results: list[SearchResult],
        intent: QueryIntent,
        request: LiteratureRequest,
    ) -> list[DocumentCandidate]:
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for result in chunk_results:
            grouped[result.chunk.document_id].append(result)

        candidates = []
        recall_limit = max(request.top_k_documents * 4, 20)
        for document in self.store.documents.values():
            document_results = grouped.get(document.document_id, [])
            score = self._document_recall_score(document.document_id, document_results, intent)
            if score <= 0:
                continue
            candidates.append(
                DocumentCandidate(
                    document=document,
                    results=document_results,
                    score=score,
                    topic_score=self._document_topic_score(document.document_id, document_results, intent),
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:recall_limit]

    def _dedupe_document_candidates(
        self,
        candidates: list[DocumentCandidate],
    ) -> list[DocumentCandidate]:
        by_key: dict[str, DocumentCandidate] = {}
        for candidate in candidates:
            key = self._document_identity_key(candidate.document)
            existing = by_key.get(key)
            if existing is None or self._prefer_candidate(candidate, existing):
                by_key[key] = candidate
        return sorted(by_key.values(), key=lambda item: item.score, reverse=True)

    def _document_identity_key(self, document: Document) -> str:
        doi = document.metadata.doi.strip().lower() if document.metadata.doi else ""
        if doi:
            return f"doi:{doi}"
        title = self._normalize_topic_text(document.metadata.title or "")
        if title:
            return f"title:{title}"
        return f"filename:{document.filename.lower()}"

    def _prefer_candidate(self, candidate: DocumentCandidate, existing: DocumentCandidate) -> bool:
        candidate_is_duplicate = bool(candidate.document.metadata.duplicate_of)
        existing_is_duplicate = bool(existing.document.metadata.duplicate_of)
        if candidate_is_duplicate != existing_is_duplicate:
            return not candidate_is_duplicate
        return candidate.score > existing.score

    def _rerank_document_candidates(
        self,
        candidates: list[DocumentCandidate],
        intent: QueryIntent,
        request: LiteratureRequest,
    ) -> tuple[list[DocumentCandidate], str, str | None]:
        if len(candidates) <= 1:
            return candidates, "local_topic_metadata_rerank", None
        if getattr(self.answerer, "client", None) is None:
            return candidates, "local_topic_metadata_rerank", None

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._call_reranker_llm, candidates[:20], intent, request)
            payload = future.result(timeout=self._reranker_timeout())
        except FuturesTimeoutError:
            return candidates, "local_topic_metadata_rerank", "LLM reranker timed out; fell back to local ranking."
        except Exception as exc:
            return candidates, "local_topic_metadata_rerank", f"LLM reranker failed: {type(exc).__name__}"
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        score_map = {}
        reason_map = {}
        for item in payload.get("rankings", []):
            if not isinstance(item, dict):
                continue
            document_id = str(item.get("document_id", ""))
            if not document_id:
                continue
            try:
                relevance = float(item.get("relevance", 0.0))
            except (TypeError, ValueError):
                relevance = 0.0
            score_map[document_id] = max(0.0, min(1.0, relevance))
            reason_map[document_id] = self._shorten(str(item.get("reason", "")).strip(), 180) or None

        if not score_map:
            return candidates, "local_topic_metadata_rerank", "LLM reranker returned no usable rankings."

        reranked = []
        for candidate in candidates:
            relevance = score_map.get(candidate.document.document_id, 0.0)
            reason = reason_map.get(candidate.document.document_id)
            reranked.append(
                DocumentCandidate(
                    document=candidate.document,
                    results=candidate.results,
                    score=candidate.score + relevance * 4.0,
                    topic_score=candidate.topic_score,
                    rerank_reason=reason,
                )
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True), "llm_candidate_rerank", None

    def _call_reranker_llm(
        self,
        candidates: list[DocumentCandidate],
        intent: QueryIntent,
        request: LiteratureRequest,
    ) -> dict:
        papers = []
        for candidate in candidates:
            metadata = candidate.document.metadata
            papers.append(
                {
                    "document_id": candidate.document.document_id,
                    "title": metadata.title or candidate.document.filename,
                    "abstract": self._shorten(metadata.abstract or "", 700),
                    "keywords": metadata.keywords[:12],
                    "fields_of_study": metadata.fields_of_study[:8],
                    "local_score": round(candidate.score, 4),
                }
            )
        prompt = (
            "你是论文相关性 reranker。请根据用户研究方向判断候选论文是否是该方向的已有文献，"
            "不要把其他领域的相似方法当作相关论文。输出严格 JSON，不要 Markdown。\n"
            "JSON 字段：rankings；每项包含 document_id, relevance(0到1), reason。\n\n"
            f"研究方向：{request.query}\n"
            f"关注重点：{request.focus or ''}\n"
            f"必需主题组：{intent.required_groups}\n"
            f"排除主题：{intent.exclude_terms}\n"
            f"候选论文：{json.dumps(papers, ensure_ascii=False)}"
        )
        output = self.answerer.complete(
            prompt,
            system=(
                "You are a strict paper relevance reranker. "
                "Return one valid JSON object only. Do not summarize papers."
            ),
        )
        return self._parse_json_object(output)

    def _document_recall_score(
        self,
        document_id: str,
        results: list[SearchResult],
        intent: QueryIntent,
    ) -> float:
        document = self.store.documents.get(document_id)
        if document is None:
            return 0.0

        sorted_results = sorted(results, key=lambda item: item.score, reverse=True)
        chunk_score = sum(item.score for item in sorted_results[:5])
        topic_score = self._document_topic_score(document_id, results, intent)
        metadata_text = self._normalize_topic_text(
            " ".join(
                item
                for item in [
                    document.metadata.title,
                    document.metadata.abstract,
                    " ".join(document.metadata.keywords),
                    " ".join(document.metadata.fields_of_study),
                ]
                if item
            )
        )
        metadata_hits = sum(1 for term in intent.relevance_terms if term in metadata_text)
        group_hits = sum(
            1
            for group in intent.required_groups
            if any(term in metadata_text for term in group)
        )
        exclude_hits = sum(1 for term in intent.exclude_terms if term in metadata_text)
        score = chunk_score + topic_score * 0.35 + metadata_hits * 0.25 + group_hits * 0.6
        score -= exclude_hits * 1.2
        if document.metadata.duplicate_of:
            score *= 0.85
        return float(score)

    def _extract_evidence(
        self,
        candidates: list[DocumentCandidate],
        intent: QueryIntent,
        request: LiteratureRequest,
    ) -> dict[str, list[SearchResult]]:
        if not candidates:
            return {}

        per_document_limit = max(1, min(request.evidence_k, request.evidence_k // len(candidates) or 1))
        scored_by_document: dict[str, list[SearchResult]] = {}
        pool: list[SearchResult] = []
        for candidate in candidates:
            document_id = candidate.document.document_id
            existing_scores = {result.chunk.chunk_id: result.score for result in candidate.results}
            chunks = self._document_chunks(document_id, request.section_filter)
            if not chunks and request.section_filter:
                chunks = self._document_chunks(document_id, None)

            scored_results = []
            for chunk in chunks:
                local_score = self._chunk_relevance_score(chunk.text, intent)
                score = existing_scores.get(chunk.chunk_id, 0.0) + local_score * 0.08 + candidate.topic_score * 0.01
                if score > 0:
                    scored_results.append(SearchResult(chunk=chunk, score=float(score)))

            if not scored_results and chunks:
                scored_results.append(SearchResult(chunk=chunks[0], score=0.01))

            scored_results = sorted(scored_results, key=lambda item: item.score, reverse=True)
            scored_by_document[document_id] = scored_results[:per_document_limit]
            pool.extend(scored_results[per_document_limit:])

        selected_chunk_ids = {
            result.chunk.chunk_id
            for results in scored_by_document.values()
            for result in results
        }
        total_selected = sum(len(results) for results in scored_by_document.values())
        for result in sorted(pool, key=lambda item: item.score, reverse=True):
            if total_selected >= request.evidence_k:
                break
            if result.chunk.chunk_id in selected_chunk_ids:
                continue
            scored_by_document[result.chunk.document_id].append(result)
            selected_chunk_ids.add(result.chunk.chunk_id)
            total_selected += 1

        return scored_by_document

    def _document_chunks(self, document_id: str, section_filter: str | None) -> list:
        section = section_filter.strip().lower() if section_filter else None
        chunks = [chunk for chunk in self.store.chunks if chunk.document_id == document_id]
        if not section:
            return chunks
        return [chunk for chunk in chunks if chunk.section == section]

    def _chunk_relevance_score(self, text: str, intent: QueryIntent) -> int:
        normalized = self._normalize_topic_text(text)
        score = sum(1 for term in intent.relevance_terms if term in normalized)
        score += sum(1 for group in intent.required_groups if any(term in normalized for term in group))
        return score

    def _paper_candidate(
        self,
        candidate: DocumentCandidate,
        evidence_results: list[SearchResult],
    ) -> PaperCandidate:
        document = candidate.document
        sorted_results = sorted(evidence_results, key=lambda item: item.score, reverse=True)
        pages = sorted({item.chunk.page for item in sorted_results})
        sections = sorted({item.chunk.section for item in sorted_results if item.chunk.section})
        preview_source = ""
        if sorted_results:
            preview_source = sorted_results[0].chunk.text
        else:
            preview_source = document.metadata.abstract or document.metadata.title or document.filename
        score = candidate.score + sum(item.score for item in sorted_results[:3]) * 0.05
        return PaperCandidate(
            document_id=document.document_id,
            filename=document.filename,
            pages=document.pages,
            chunks=document.chunks,
            metadata=paper_metadata(document.metadata),
            score=round(float(score), 6),
            evidence_count=len(evidence_results),
            evidence_pages=pages[:8],
            evidence_sections=sections[:8],
            preview=self._shorten(preview_source, max_chars=260),
            rerank_reason=candidate.rerank_reason,
        )

    def _document_matches_topic(
        self,
        document_id: str,
        results: list[SearchResult],
        intent: QueryIntent,
    ) -> bool:
        text = self._document_topic_text(document_id, results)
        has_excluded_topic = intent.exclude_terms and any(term in text for term in intent.exclude_terms)
        if intent.required_groups:
            core_groups = [group for group in intent.required_groups if not self._is_generic_task_group(group)]
            if core_groups:
                mulberry_groups = [group for group in core_groups if "mulberry" in group]
                if mulberry_groups:
                    return all(any(term in text for term in group) for group in core_groups)
                return all(any(term in text for term in group) for group in core_groups)
            if has_excluded_topic:
                return False
            matched_groups = sum(1 for group in intent.required_groups if any(term in text for term in group))
            return matched_groups >= max(1, min(2, len(intent.required_groups)))
        if has_excluded_topic:
            return False
        return self._document_topic_score(document_id, results, intent) >= 2

    def _is_generic_task_group(self, group: list[str]) -> bool:
        generic_terms = {
            "检测",
            "识别",
            "分类",
            "detection",
            "recognition",
            "classification",
        }
        return any(term in generic_terms for term in group)

    def _document_topic_score(
        self,
        document_id: str,
        results: list[SearchResult],
        intent: QueryIntent,
    ) -> int:
        text = self._document_topic_text(document_id, results)
        score = sum(1 for term in intent.relevance_terms if term in text)
        score += sum(1 for group in intent.required_groups if any(term in text for term in group))
        score -= sum(1 for term in intent.exclude_terms if term in text)
        return score

    def _document_topic_text(self, document_id: str, results: list[SearchResult]) -> str:
        document = self.store.documents.get(document_id)
        if document is None:
            return ""
        all_chunks = [chunk.text for chunk in self.store.chunks if chunk.document_id == document_id][:12]
        result_chunks = [result.chunk.text for result in results[:8]]
        return self._normalize_topic_text(
            " ".join(
                item
                for item in [
                    document.filename,
                    document.metadata.title,
                    document.metadata.abstract,
                    document.metadata.venue,
                    " ".join(document.metadata.keywords),
                    " ".join(document.metadata.fields_of_study),
                    " ".join(result_chunks),
                    " ".join(all_chunks),
                ]
                if item
            )
        )

    def _topic_tokens(self, value: str) -> list[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "for",
            "in",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
            "using",
            "研究",
            "方法",
            "论文",
            "综述",
            "应用",
            "方向",
        }
        return self._unique_terms(
            [
                word
                for word in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", value.lower())
                if word not in stop_words and len(word) >= 2
            ]
        )

    def _unique_terms(self, terms: list[str]) -> list[str]:
        unique = []
        for term in terms:
            normalized = self._normalize_topic_text(term)
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique

    def _normalize_topic_text(self, value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", value.lower()))

    def _excluded_candidate_titles(
        self,
        recalled_candidates: list[DocumentCandidate],
        gated_candidates: list[DocumentCandidate],
    ) -> list[str]:
        gated_ids = {candidate.document.document_id for candidate in gated_candidates}
        titles = []
        for candidate in recalled_candidates:
            if candidate.document.document_id in gated_ids:
                continue
            title = candidate.document.metadata.title or candidate.document.filename
            if title not in titles:
                titles.append(title)
        return titles

    def _evidence_coverage(self, intent: QueryIntent, results: list[SearchResult]) -> float:
        if not results:
            return 0.0
        text = self._normalize_topic_text(" ".join(result.chunk.text for result in results))
        core_groups = [group for group in intent.required_groups if not self._is_generic_task_group(group)]
        if core_groups:
            matched = sum(1 for group in core_groups if any(term in text for term in group))
            return round(matched / len(core_groups), 4)
        important_terms = intent.relevance_terms[:8]
        if not important_terms:
            return 1.0
        matched_terms = sum(1 for term in important_terms if term in text)
        return round(matched_terms / len(important_terms), 4)

    def _requires_evidence_guard(self, trace: LiteratureRetrievalTrace) -> bool:
        core_groups = [
            group
            for group in trace.required_groups
            if not self._is_generic_task_group(group)
        ]
        return bool(core_groups)

    def _shorten(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
