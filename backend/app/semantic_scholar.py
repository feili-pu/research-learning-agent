import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass
class SemanticScholarWork:
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    publisher: str | None = None
    external_url: str | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    fields_of_study: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    confidence: str = "low"
    match_score: float = 0.0


class SemanticScholarClient:
    def __init__(self, base_url: str = "https://api.semanticscholar.org/graph/v1", timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search_by_title(self, title: str) -> SemanticScholarWork | None:
        clean_title = title.strip()
        if not clean_title:
            return None

        params = urlencode(
            {
                "query": clean_title,
                "limit": "5",
                "fields": ",".join(
                    [
                        "title",
                        "authors",
                        "year",
                        "venue",
                        "abstract",
                        "citationCount",
                        "referenceCount",
                        "url",
                        "fieldsOfStudy",
                        "externalIds",
                    ]
                ),
            }
        )
        request = Request(
            f"{self.base_url}/paper/search?{params}",
            headers={
                "Accept": "application/json",
                "User-Agent": "research-learning-agent/0.10",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return None

        papers = payload.get("data", [])
        if not isinstance(papers, list) or not papers:
            return None

        best_paper = None
        best_score = 0.0
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            candidate_title = paper.get("title")
            if not candidate_title:
                continue
            score = self._title_similarity(clean_title, candidate_title)
            if score > best_score:
                best_score = score
                best_paper = paper

        if best_paper is None or best_score < 0.72:
            return None

        return self._paper_to_work(best_paper, best_score)

    def _paper_to_work(self, paper: dict, score: float) -> SemanticScholarWork:
        external_ids = paper.get("externalIds") or {}
        fields = [item for item in paper.get("fieldsOfStudy") or [] if isinstance(item, str)]
        return SemanticScholarWork(
            title=paper.get("title"),
            authors=self._authors(paper.get("authors", [])),
            year=paper.get("year"),
            venue=paper.get("venue"),
            doi=external_ids.get("DOI") if isinstance(external_ids, dict) else None,
            abstract=paper.get("abstract"),
            external_url=paper.get("url"),
            reference_count=paper.get("referenceCount"),
            citation_count=paper.get("citationCount"),
            fields_of_study=fields[:12],
            keywords=fields[:12],
            confidence=self._confidence(score),
            match_score=round(score, 4),
        )

    def _authors(self, authors) -> str | None:
        if not isinstance(authors, list):
            return None
        names = [author.get("name") for author in authors[:20] if isinstance(author, dict) and author.get("name")]
        return ", ".join(names) if names else None

    def _title_similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(None, self._normalize_title(left), self._normalize_title(right)).ratio()

    def _normalize_title(self, title: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()

    def _confidence(self, score: float) -> str:
        if score >= 0.95:
            return "high"
        if score >= 0.84:
            return "medium"
        return "low"
