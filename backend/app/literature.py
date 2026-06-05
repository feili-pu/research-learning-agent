from collections import defaultdict

from .answerer import Answerer
from .rag import RagStore, SearchResult
from .schemas import (
    LiteratureRequest,
    LiteratureReviewResponse,
    LiteratureSearchResponse,
    PaperCandidate,
    SourceChunk,
)


class LiteratureService:
    def __init__(self, store: RagStore, answerer: Answerer) -> None:
        self.store = store
        self.answerer = answerer

    def search(self, request: LiteratureRequest) -> LiteratureSearchResponse:
        results = self.store.search(self._search_query(request), request.evidence_k, request.section_filter)
        papers = self._rank_papers(results, request.top_k_documents)
        filtered_results = self._filter_results_to_papers(results, papers)
        return LiteratureSearchResponse(
            query=request.query,
            retrieval_mode=self.store.active_retrieval_mode,
            papers=papers,
            sources=self._source_chunks(filtered_results),
        )

    def review(self, request: LiteratureRequest) -> LiteratureReviewResponse:
        prompt = self._build_prompt(
            task="direction_review",
            request=request,
            instruction=(
                "请把这些论文当作一个后台论文库来使用，围绕用户给定研究方向生成中文文献综述。"
                "输出应包含：研究背景、核心问题、代表论文、方法分类、主要结论、局限性、可继续深入的问题。"
                "不要试图总结整个论文库，只总结与研究方向相关的论文。"
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
            ),
        )
        return self._run("detail_briefing", request, prompt)

    def _run(
        self,
        task: str,
        request: LiteratureRequest,
        prompt: str,
    ) -> LiteratureReviewResponse:
        results = self.store.search(self._search_query(request), request.evidence_k, request.section_filter)
        papers = self._rank_papers(results, request.top_k_documents)
        filtered_results = self._filter_results_to_papers(results, papers)
        answer = self.answerer.answer(prompt, filtered_results)
        return LiteratureReviewResponse(
            task=task,
            query=request.query,
            retrieval_mode=self.store.active_retrieval_mode,
            answer_mode=answer.answer_mode,
            model=answer.model,
            answer=answer.answer,
            papers=papers,
            sources=self._source_chunks(filtered_results),
        )

    def _search_query(self, request: LiteratureRequest) -> str:
        if not request.focus:
            return request.query
        return f"{request.query}\n关注重点：{request.focus}"

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
        )

    def _rank_papers(
        self,
        results: list[SearchResult],
        top_k_documents: int,
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
            score = sum(item.score for item in sorted_results[:3])
            pages = sorted({item.chunk.page for item in sorted_results})
            sections = sorted({item.chunk.section for item in sorted_results if item.chunk.section})
            preview = self._shorten(sorted_results[0].chunk.text, max_chars=260)
            candidates.append(
                PaperCandidate(
                    document_id=document.document_id,
                    filename=document.filename,
                    pages=document.pages,
                    chunks=document.chunks,
                    metadata=self._paper_metadata(document),
                    score=round(float(score), 6),
                    evidence_count=len(document_results),
                    evidence_pages=pages[:8],
                    evidence_sections=sections[:8],
                    preview=preview,
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k_documents]

    def _filter_results_to_papers(
        self,
        results: list[SearchResult],
        papers: list[PaperCandidate],
    ) -> list[SearchResult]:
        allowed_ids = {paper.document_id for paper in papers}
        return [result for result in results if result.chunk.document_id in allowed_ids]

    def _source_chunks(self, results: list[SearchResult]) -> list[SourceChunk]:
        return [
            SourceChunk(
                document_id=result.chunk.document_id,
                filename=result.chunk.filename,
                page=result.chunk.page,
                chunk_id=result.chunk.chunk_id,
                score=result.score,
                text=result.chunk.text,
                section=result.chunk.section,
            )
            for result in results
        ]

    def _shorten(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _paper_metadata(self, document) -> dict:
        return {
            "title": document.metadata.title,
            "authors": document.metadata.authors,
            "year": document.metadata.year,
            "venue": document.metadata.venue,
            "doi": document.metadata.doi,
            "abstract": document.metadata.abstract,
            "publisher": document.metadata.publisher,
            "external_url": document.metadata.external_url,
            "reference_count": document.metadata.reference_count,
            "citation_count": document.metadata.citation_count,
            "fields_of_study": document.metadata.fields_of_study,
            "metadata_confidence": document.metadata.metadata_confidence,
            "metadata_match_score": document.metadata.metadata_match_score,
            "metadata_source": document.metadata.metadata_source,
            "is_enriched": document.metadata.is_enriched,
            "keywords": document.metadata.keywords,
            "duplicate_of": document.metadata.duplicate_of,
            "duplicate_reason": document.metadata.duplicate_reason,
        }
