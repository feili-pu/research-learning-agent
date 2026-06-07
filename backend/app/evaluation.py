from dataclasses import dataclass

from .literature import LiteratureService
from .schemas import (
    EvaluationCaseResult,
    LiteratureEvaluationRequest,
    LiteratureEvaluationResponse,
    LiteratureRequest,
)


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    query: str
    focus: str | None
    expected_terms: tuple[str, ...]
    section_filter: str | None = None
    forbidden_terms: tuple[str, ...] = ()
    expected_titles: tuple[str, ...] = ()
    forbidden_titles: tuple[str, ...] = ()
    min_recall: float = 0.6
    max_noise: float = 0.0


DEFAULT_CASES = [
    EvaluationCase(
        name="water_quality_methods",
        query="water quality prediction",
        focus="models, methods, and experiments",
        expected_terms=("water", "quality", "model"),
        section_filter="methods",
        forbidden_terms=("biometric", "mulberry", "recommendation"),
    ),
    EvaluationCase(
        name="biometric_security",
        query="biometric security template protection",
        focus="methods and security limitations",
        expected_terms=("biometric", "security", "template"),
        section_filter=None,
        forbidden_terms=("water quality", "mulberry", "recommender"),
    ),
    EvaluationCase(
        name="remote_sensing_water_quality",
        query="remote sensing water quality UAV hyperspectral",
        focus="data sources and retrieval methods",
        expected_terms=("remote", "sensing", "water"),
        section_filter=None,
        forbidden_terms=("biometric", "mulberry", "recommendation"),
    ),
    EvaluationCase(
        name="mulberry_leaf_disease_detection",
        query="mulberry leaf disease detection",
        focus="deep learning methods and experiments",
        expected_terms=("mulberry", "leaf", "disease"),
        expected_titles=("mulberry leaf disease",),
        forbidden_terms=("water quality", "biometric", "template protection", "recommendation"),
        forbidden_titles=("Water Quality", "Biometric", "Template Protection", "Recommender"),
        min_recall=0.6,
    ),
]


class EvaluationService:
    def __init__(self, literature_service: LiteratureService) -> None:
        self.literature_service = literature_service

    def evaluate_literature(
        self,
        request: LiteratureEvaluationRequest,
        cases: list[EvaluationCase] | None = None,
    ) -> LiteratureEvaluationResponse:
        active_cases = cases or DEFAULT_CASES
        results = [self._evaluate_case(case, request) for case in active_cases]
        passed_cases = sum(1 for result in results if result.passed)
        average_score = sum(result.score for result in results) / len(results) if results else 0.0
        return LiteratureEvaluationResponse(
            retrieval_mode=self.literature_service.store.active_retrieval_mode,
            total_cases=len(results),
            passed_cases=passed_cases,
            average_score=round(average_score, 4),
            cases=results,
        )

    def _evaluate_case(
        self,
        case: EvaluationCase,
        request: LiteratureEvaluationRequest,
    ) -> EvaluationCaseResult:
        section_filter = request.section_filter or case.section_filter
        search_response = self.literature_service.search(
            LiteratureRequest(
                query=case.query,
                focus=case.focus,
                top_k_documents=request.top_k_documents,
                evidence_k=request.evidence_k,
                section_filter=section_filter,
            )
        )
        evidence_text = self._join_evidence(search_response)
        title_text = self._join_titles(search_response)
        matched_terms = [
            term for term in case.expected_terms if self._matches(term, evidence_text)
        ]
        missing_terms = [
            term for term in case.expected_terms if not self._matches(term, evidence_text)
        ]
        forbidden_hits = [
            term for term in case.forbidden_terms if self._matches(term, evidence_text)
        ]
        matched_titles = [
            title for title in case.expected_titles if self._matches(title, title_text)
        ]
        missing_titles = [
            title for title in case.expected_titles if not self._matches(title, title_text)
        ]
        forbidden_title_hits = [
            title for title in case.forbidden_titles if self._matches(title, title_text)
        ]
        recall = len(matched_terms) / len(case.expected_terms) if case.expected_terms else 1.0
        title_recall = len(matched_titles) / len(case.expected_titles) if case.expected_titles else recall
        noise_denominator = max(len(case.forbidden_terms) + len(case.forbidden_titles), 1)
        noise = (len(forbidden_hits) + len(forbidden_title_hits)) / noise_denominator
        precision = 1.0 - noise
        score = max(0.0, ((recall + title_recall + precision) / 3) - noise * 0.35)
        return EvaluationCaseResult(
            name=case.name,
            query=case.query,
            focus=case.focus,
            section_filter=section_filter,
            expected_terms=list(case.expected_terms),
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            forbidden_terms=list(case.forbidden_terms),
            forbidden_hits=forbidden_hits,
            expected_titles=list(case.expected_titles),
            matched_titles=matched_titles,
            missing_titles=missing_titles,
            forbidden_titles=list(case.forbidden_titles),
            forbidden_title_hits=forbidden_title_hits,
            precision=round(precision, 4),
            recall=round(recall, 4),
            noise=round(noise, 4),
            score=round(score, 4),
            passed=recall >= case.min_recall and noise <= case.max_noise and not missing_titles,
            papers=search_response.papers,
            sources=search_response.sources,
        )

    def _join_evidence(self, search_response) -> str:
        parts = []
        for paper in search_response.papers:
            metadata = paper.metadata
            parts.extend(
                [
                    paper.filename,
                    metadata.title or "",
                    metadata.authors or "",
                    metadata.venue or "",
                    metadata.abstract or "",
                    " ".join(metadata.keywords),
                    paper.preview,
                ]
            )
        parts.extend(source.text for source in search_response.sources)
        return " ".join(parts).lower()

    def _join_titles(self, search_response) -> str:
        return " ".join(
            paper.metadata.title or paper.filename
            for paper in search_response.papers
        ).lower()

    def _matches(self, needle: str, text: str) -> bool:
        return needle.lower() in text
