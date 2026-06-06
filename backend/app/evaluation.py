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


DEFAULT_CASES = [
    EvaluationCase(
        name="water_quality_methods",
        query="water quality prediction",
        focus="models, methods, and experiments",
        expected_terms=("water", "quality", "model"),
        section_filter="methods",
    ),
    EvaluationCase(
        name="biometric_security",
        query="biometric security template protection",
        focus="methods and security limitations",
        expected_terms=("biometric", "security", "template"),
        section_filter=None,
    ),
    EvaluationCase(
        name="remote_sensing_water_quality",
        query="remote sensing water quality UAV hyperspectral",
        focus="data sources and retrieval methods",
        expected_terms=("remote", "sensing", "water"),
        section_filter=None,
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
        matched_terms = [
            term for term in case.expected_terms if term.lower() in evidence_text
        ]
        missing_terms = [
            term for term in case.expected_terms if term.lower() not in evidence_text
        ]
        score = len(matched_terms) / len(case.expected_terms) if case.expected_terms else 0.0
        return EvaluationCaseResult(
            name=case.name,
            query=case.query,
            focus=case.focus,
            section_filter=section_filter,
            expected_terms=list(case.expected_terms),
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            score=round(score, 4),
            passed=score >= 0.6,
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
