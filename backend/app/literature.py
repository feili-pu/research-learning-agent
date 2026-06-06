from collections import defaultdict
from dataclasses import dataclass, field
import re

from .answerer import Answerer
from .presenters import paper_metadata, source_chunks
from .rag import RagStore, SearchResult
from .schemas import (
    LiteratureRequest,
    LiteratureReviewResponse,
    LiteratureSearchResponse,
    PaperCandidate,
)


@dataclass(frozen=True)
class QueryIntent:
    search_query: str
    required_groups: list[list[str]] = field(default_factory=list)
    relevance_terms: list[str] = field(default_factory=list)


class LiteratureService:
    def __init__(self, store: RagStore, answerer: Answerer) -> None:
        self.store = store
        self.answerer = answerer

    def search(self, request: LiteratureRequest) -> LiteratureSearchResponse:
        intent = self._query_intent(request)
        results = self.store.search(
            intent.search_query,
            self._candidate_evidence_k(request),
            request.section_filter,
            allow_semantic=False,
        )
        results = self._filter_results_by_topic(results, intent)
        papers = self._rank_papers(results, request.top_k_documents, intent)
        filtered_results = self._filter_results_to_papers(results, papers)
        return LiteratureSearchResponse(
            query=request.query,
            retrieval_mode=self.store.active_retrieval_mode,
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
        intent = self._query_intent(request)
        results = self.store.search(
            intent.search_query,
            self._candidate_evidence_k(request),
            request.section_filter,
            allow_semantic=False,
        )
        results = self._filter_results_by_topic(results, intent)
        papers = self._rank_papers(results, request.top_k_documents, intent)
        filtered_results = self._filter_results_to_papers(results, papers)
        filtered_results = filtered_results[: request.evidence_k]
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
                papers=[],
                sources=[],
            )
        answer = self.answerer.answer(prompt, filtered_results)
        return LiteratureReviewResponse(
            task=task,
            query=request.query,
            retrieval_mode=self.store.active_retrieval_mode,
            answer_mode=answer.answer_mode,
            model=answer.model,
            answer=answer.answer,
            papers=papers,
            sources=source_chunks(filtered_results),
        )

    def _candidate_evidence_k(self, request: LiteratureRequest) -> int:
        return min(max(request.evidence_k * 4, request.evidence_k), 80)

    def _search_query(self, request: LiteratureRequest) -> str:
        if not request.focus:
            return request.query
        return f"{request.query}\n关注重点：{request.focus}"

    def _query_intent(self, request: LiteratureRequest) -> QueryIntent:
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
        return QueryIntent(
            search_query=" ".join(self._unique_terms(search_terms)) or raw_query,
            required_groups=self._dedupe_groups(required_groups),
            relevance_terms=relevance_terms,
        )

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

    def _rank_papers(
        self,
        results: list[SearchResult],
        top_k_documents: int,
        intent: QueryIntent,
    ) -> list[PaperCandidate]:
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for result in results:
            grouped[result.chunk.document_id].append(result)

        candidates = []
        for document_id, document_results in grouped.items():
            document = self.store.documents.get(document_id)
            if document is None:
                continue

            sorted_results = sorted(document_results, key=lambda item: item.score, reverse=True)
            topic_score = self._document_topic_score(document_id, document_results, intent)
            score = sum(item.score for item in sorted_results[:3]) + topic_score * 0.25
            pages = sorted({item.chunk.page for item in sorted_results})
            sections = sorted({item.chunk.section for item in sorted_results if item.chunk.section})
            preview = self._shorten(sorted_results[0].chunk.text, max_chars=260)
            candidates.append(
                PaperCandidate(
                    document_id=document.document_id,
                    filename=document.filename,
                    pages=document.pages,
                    chunks=document.chunks,
                    metadata=paper_metadata(document.metadata),
                    score=round(float(score), 6),
                    evidence_count=len(document_results),
                    evidence_pages=pages[:8],
                    evidence_sections=sections[:8],
                    preview=preview,
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k_documents]

    def _filter_results_by_topic(
        self,
        results: list[SearchResult],
        intent: QueryIntent,
    ) -> list[SearchResult]:
        if not intent.required_groups and not intent.relevance_terms:
            return results
        grouped: dict[str, list[SearchResult]] = defaultdict(list)
        for result in results:
            grouped[result.chunk.document_id].append(result)

        allowed_ids = {
            document_id
            for document_id, document_results in grouped.items()
            if self._document_matches_topic(document_id, document_results, intent)
        }
        return [result for result in results if result.chunk.document_id in allowed_ids]

    def _document_matches_topic(
        self,
        document_id: str,
        results: list[SearchResult],
        intent: QueryIntent,
    ) -> bool:
        text = self._document_topic_text(document_id, results)
        if intent.required_groups:
            core_groups = [group for group in intent.required_groups if not self._is_generic_task_group(group)]
            if core_groups:
                mulberry_groups = [group for group in core_groups if "mulberry" in group]
                if mulberry_groups:
                    return any(
                        any(term in text for term in group)
                        for group in mulberry_groups
                    ) and self._document_topic_score(document_id, results, intent) >= 2
                return all(any(term in text for term in group) for group in core_groups)
            matched_groups = sum(1 for group in intent.required_groups if any(term in text for term in group))
            return matched_groups >= max(1, min(2, len(intent.required_groups)))
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

    def _filter_results_to_papers(
        self,
        results: list[SearchResult],
        papers: list[PaperCandidate],
    ) -> list[SearchResult]:
        allowed_ids = {paper.document_id for paper in papers}
        return [result for result in results if result.chunk.document_id in allowed_ids]

    def _shorten(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
