import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import hashlib
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .rag import PaperMetadata, RagStore
from .schemas import DiscoveryPaper, DiscoveryRequest, DiscoveryResponse


@dataclass
class ProviderResult:
    source: str
    source_id: str | None
    title: str
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    external_url: str | None = None
    pdf_url: str | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    fields_of_study: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    is_open_access: bool = False
    relevance_score: float = 0.0


@dataclass(frozen=True)
class QueryPlan:
    raw_query: str
    queries: list[str]
    relevance_terms: list[str]
    planner: str = "llm"
    model: str | None = None
    errors: list[str] = field(default_factory=list)


DISCOVERY_SEARCH_TIMEOUT_SECONDS = 12
DISCOVERY_PLANNER_TIMEOUT_SECONDS = 20


class DiscoveryService:
    def __init__(
        self,
        store: RagStore | None = None,
        providers: dict[str, object] | None = None,
        answerer=None,
        store_provider=None,
    ) -> None:
        self.store = store
        self.store_provider = store_provider
        self.answerer = answerer
        self.providers = providers or {
            "semantic_scholar": SemanticScholarDiscoveryClient(),
            "crossref": CrossrefDiscoveryClient(),
            "arxiv": ArxivDiscoveryClient(),
            "openalex": OpenAlexDiscoveryClient(),
        }

    def search(self, request: DiscoveryRequest) -> DiscoveryResponse:
        source_names = self._normalize_sources(request.sources)
        query_plan = self._build_query_plan(request)
        papers: list[DiscoveryPaper] = []
        errors: list[str] = list(query_plan.errors)

        if not query_plan.queries:
            return DiscoveryResponse(
                query=request.query,
                focus=request.focus,
                sources=source_names,
                queries_used=[],
                query_planner=query_plan.planner,
                planner_model=query_plan.model,
                papers=[],
                errors=errors,
            )

        futures = {}
        task_count = len(source_names) * len(query_plan.queries)
        executor = ThreadPoolExecutor(max_workers=min(task_count, 6) or 1)
        try:
            for source in source_names:
                provider = self.providers.get(source)
                if provider is None:
                    errors.append(f"Unsupported discovery source: {source}")
                    continue
                for planned_query in query_plan.queries:
                    futures[executor.submit(provider.search, planned_query, request.limit_per_source)] = source

            try:
                completed_futures = as_completed(futures, timeout=DISCOVERY_SEARCH_TIMEOUT_SECONDS)
                for future in completed_futures:
                    source = futures[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        errors.append(f"{source}: {type(exc).__name__}")
                        continue
                    for result in results:
                        if not result.title:
                            continue
                        paper = self._paper_from_result(result, query_plan.relevance_terms)
                        if self._is_relevant(paper, query_plan.relevance_terms):
                            papers.append(paper)
            except FuturesTimeoutError:
                pending_sources = sorted({source for future, source in futures.items() if not future.done()})
                if pending_sources:
                    errors.append(f"Timed out while searching: {', '.join(pending_sources)}")
                for future in futures:
                    if not future.done():
                        future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return DiscoveryResponse(
            query=request.query,
            focus=request.focus,
            sources=source_names,
            queries_used=query_plan.queries,
            query_planner=query_plan.planner,
            planner_model=query_plan.model,
            papers=self._dedupe_and_rank(papers),
            errors=errors,
        )

    def import_metadata(self, paper: DiscoveryPaper):
        metadata = PaperMetadata(
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            doi=paper.doi,
            abstract=paper.abstract,
            external_url=paper.external_url,
            reference_count=paper.reference_count,
            citation_count=paper.citation_count,
            fields_of_study=paper.fields_of_study,
            metadata_confidence="medium",
            metadata_match_score=paper.relevance_score,
            metadata_source=paper.source,
            is_enriched=True,
            keywords=paper.keywords,
        )
        filename = self._metadata_filename(paper)
        return self._get_store().add_metadata_document(filename, metadata)

    def _get_store(self) -> RagStore:
        if self.store is None:
            if self.store_provider is None:
                raise RuntimeError("DiscoveryService has no RagStore configured.")
            self.store = self.store_provider()
        return self.store

    def _search_query(self, request: DiscoveryRequest) -> str:
        if not request.focus:
            return request.query
        return f"{request.query} {request.focus}"

    def _build_query_plan(self, request: DiscoveryRequest) -> QueryPlan:
        raw_query = self._normalize_space(self._search_query(request))
        if self.answerer is None or getattr(self.answerer, "client", None) is None:
            return QueryPlan(
                raw_query=raw_query,
                queries=[],
                relevance_terms=[],
                model=getattr(self.answerer, "model", None),
                errors=["LLM query planning is required for literature discovery. Configure OPENAI_API_KEY first."],
            )

        planner_executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = planner_executor.submit(self._call_query_planner_llm, request, raw_query)
            payload = future.result(timeout=DISCOVERY_PLANNER_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            future.cancel()
            return QueryPlan(
                raw_query=raw_query,
                queries=[],
                relevance_terms=[],
                model=getattr(self.answerer, "model", None),
                errors=["LLM query planning timed out. Check the model, API key, base URL, or relay service."],
            )
        except Exception as exc:
            return QueryPlan(
                raw_query=raw_query,
                queries=[],
                relevance_terms=[],
                model=getattr(self.answerer, "model", None),
                errors=[f"LLM query planning failed: {type(exc).__name__}"],
            )
        finally:
            planner_executor.shutdown(wait=False, cancel_futures=True)

        queries = self._unique_queries([str(item) for item in payload.get("queries", [])])[:3]
        relevance_terms = self._unique_items(
            [
                *[str(item) for item in payload.get("relevance_terms", [])],
                *self._query_terms(" ".join([raw_query, *queries])),
            ]
        )[:18]
        if not queries:
            return QueryPlan(
                raw_query=raw_query,
                queries=[],
                relevance_terms=[],
                model=getattr(self.answerer, "model", None),
                errors=["LLM query planning returned no usable search queries."],
            )
        return QueryPlan(
            raw_query=raw_query,
            queries=queries,
            relevance_terms=relevance_terms,
            model=getattr(self.answerer, "model", None),
        )

    def _call_query_planner_llm(self, request: DiscoveryRequest, raw_query: str) -> dict:
        system_prompt = (
            "You are an expert academic literature search strategist. "
            "Convert the user's Chinese or English research intent into precise external paper database queries. "
            "Return only valid JSON. Do not include markdown."
        )
        user_prompt = "\n".join(
            [
                "Create a literature discovery query plan.",
                f"Research query: {request.query}",
                f"Focus: {request.focus or ''}",
                "Requirements:",
                "- Produce 2 to 3 concise English academic search queries for Semantic Scholar, OpenAlex, Crossref, or arXiv.",
                "- Preserve the user's core topic; do not broaden to unrelated fields.",
                "- Expand Chinese terms, acronyms, and method names into standard English academic terminology.",
                "- Include important synonyms as relevance_terms, but keep each query short.",
                "- Output JSON with this shape:",
                '{"queries":["..."],"relevance_terms":["..."]}',
                f"Combined intent: {raw_query}",
            ]
        )

        if getattr(self.answerer, "wire_api", "responses") == "chat":
            response = self.answerer.client.chat.completions.create(
                model=self.answerer.model,
                timeout=DISCOVERY_PLANNER_TIMEOUT_SECONDS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
        else:
            response = self.answerer.client.responses.create(
                model=self.answerer.model,
                timeout=DISCOVERY_PLANNER_TIMEOUT_SECONDS,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.output_text

        return self._parse_query_plan_json(content)

    def _parse_query_plan_json(self, content: str) -> dict:
        text = content.strip()
        if not text:
            raise ValueError("empty planner response")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("planner response must be a JSON object")
        return payload

    def _unique_queries(self, queries: list[str]) -> list[str]:
        return self._unique_items(
            self._normalize_space(query)
            for query in queries
            if query and self._normalize_space(query)
        )

    def _unique_items(self, items) -> list[str]:
        unique = []
        seen = set()
        for item in items:
            normalized = self._normalize_space(str(item))
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            unique.append(normalized)
            seen.add(key)
        return unique

    def _normalize_sources(self, sources: list[str]) -> list[str]:
        aliases = {
            "semantic": "semantic_scholar",
            "s2": "semantic_scholar",
            "semantic_scholar": "semantic_scholar",
            "crossref": "crossref",
            "arxiv": "arxiv",
            "openalex": "openalex",
        }
        normalized = []
        for source in sources or []:
            value = aliases.get(source.strip().lower().replace("-", "_"))
            if value and value not in normalized:
                normalized.append(value)
        return normalized or ["semantic_scholar", "openalex"]

    def _paper_from_result(self, result: ProviderResult, query_terms: list[str]) -> DiscoveryPaper:
        imported = self._find_imported_document_id(result)
        relevance_score = round((result.relevance_score * 0.35) + (self._keyword_score(result, query_terms) * 0.65), 4)
        return DiscoveryPaper(
            source=result.source,
            source_id=result.source_id,
            title=result.title,
            authors=result.authors,
            year=result.year,
            venue=result.venue,
            doi=result.doi,
            abstract=result.abstract,
            external_url=result.external_url,
            pdf_url=result.pdf_url,
            reference_count=result.reference_count,
            citation_count=result.citation_count,
            fields_of_study=result.fields_of_study,
            keywords=result.keywords,
            is_open_access=result.is_open_access,
            relevance_score=relevance_score,
            imported_document_id=imported,
        )

    def _is_relevant(self, paper: DiscoveryPaper, query_terms: list[str]) -> bool:
        if not query_terms:
            return True
        return paper.relevance_score >= 0.42 or self._has_phrase_match(paper, query_terms)

    def _keyword_score(self, result: ProviderResult, query_terms: list[str]) -> float:
        if not query_terms:
            return result.relevance_score
        text = self._paper_search_text(result)
        if not text:
            return 0.0
        matched = [term for term in query_terms if term in text]
        coverage = len(matched) / min(len(query_terms), 8)
        phrase_bonus = 0.2 if self._has_text_phrase_match(text, query_terms) else 0.0
        fuzzy_title = SequenceMatcher(None, " ".join(query_terms), self._normalize_words(result.title)).ratio()
        return min(1.0, coverage + phrase_bonus + max(0.0, fuzzy_title - 0.55) * 0.4)

    def _paper_search_text(self, result: ProviderResult) -> str:
        return self._normalize_words(
            " ".join(
                item
                for item in [
                    result.title,
                    result.abstract,
                    result.venue,
                    result.authors,
                    " ".join(result.keywords),
                    " ".join(result.fields_of_study),
                    result.doi,
                ]
                if item
            )
        )

    def _has_phrase_match(self, paper: DiscoveryPaper, query_terms: list[str]) -> bool:
        text = self._normalize_words(
            " ".join(
                item
                for item in [
                    paper.title,
                    paper.abstract,
                    paper.venue,
                    " ".join(paper.keywords),
                    " ".join(paper.fields_of_study),
                ]
                if item
            )
        )
        return self._has_text_phrase_match(text, query_terms)

    def _has_text_phrase_match(self, text: str, query_terms: list[str]) -> bool:
        for term in query_terms:
            if " " in term and term in text:
                return True
        if len(query_terms) < 2:
            return False
        joined = " ".join(query_terms)
        return joined in text

    def _query_terms(self, query: str) -> list[str]:
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
            "研究",
            "方法",
            "论文",
            "综述",
            "应用",
        }
        words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", query.lower())
        terms = []
        for word in words:
            if word in stop_words or len(word) < 2:
                continue
            if word not in terms:
                terms.append(word)
        return terms[:12]

    def _normalize_words(self, value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", value.lower()))

    def _normalize_space(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _find_imported_document_id(self, result: ProviderResult) -> str | None:
        if self.store is None:
            return None
        doi = result.doi.lower() if result.doi else None
        title = self._normalize_title(result.title)
        for document in self.store.documents.values():
            metadata = document.metadata
            if doi and metadata.doi and metadata.doi.lower() == doi:
                return document.document_id
            if title and self._normalize_title(metadata.title or "") == title:
                return document.document_id
        return None

    def _dedupe_and_rank(self, papers: list[DiscoveryPaper]) -> list[DiscoveryPaper]:
        by_key: dict[str, DiscoveryPaper] = {}
        for paper in papers:
            key = paper.doi.lower() if paper.doi else self._normalize_title(paper.title)
            if not key:
                key = f"{paper.source}:{paper.source_id or paper.title}"
            existing = by_key.get(key)
            if existing is None or paper.relevance_score > existing.relevance_score:
                by_key[key] = paper
        return sorted(
            by_key.values(),
            key=lambda item: (item.imported_document_id is None, item.relevance_score, item.citation_count or 0),
            reverse=True,
        )

    def _metadata_filename(self, paper: DiscoveryPaper) -> str:
        title = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", paper.title)
        title = re.sub(r"\s+", "-", title).strip(" .-") or "discovered-paper"
        identity = paper.doi or paper.source_id or paper.external_url or paper.title
        digest = hashlib.sha1(f"{paper.source}:{identity}".encode("utf-8")).hexdigest()[:10]
        return f"{paper.source}-{title[:80]}-{digest}.metadata"

    def _normalize_title(self, title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", title.lower())


class SemanticScholarDiscoveryClient:
    def __init__(self, base_url: str = "https://api.semanticscholar.org/graph/v1", timeout: int = 6) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, limit: int) -> list[ProviderResult]:
        params = urlencode(
            {
                "query": query,
                "limit": str(limit),
                "fields": ",".join(
                    [
                        "paperId",
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
                        "openAccessPdf",
                    ]
                ),
            }
        )
        payload = _get_json(f"{self.base_url}/paper/search?{params}", "research-learning-agent/0.15", self.timeout)
        results = []
        for index, item in enumerate(payload.get("data", []) if isinstance(payload, dict) else []):
            if not isinstance(item, dict) or not item.get("title"):
                continue
            external_ids = item.get("externalIds") or {}
            open_pdf = item.get("openAccessPdf") or {}
            fields = [value for value in item.get("fieldsOfStudy") or [] if isinstance(value, str)]
            results.append(
                ProviderResult(
                    source="semantic_scholar",
                    source_id=item.get("paperId"),
                    title=_clean_text(item.get("title")) or "",
                    authors=_authors_from_semantic_scholar(item.get("authors", [])),
                    year=item.get("year"),
                    venue=_clean_text(item.get("venue")),
                    doi=external_ids.get("DOI") if isinstance(external_ids, dict) else None,
                    abstract=_clean_text(item.get("abstract")),
                    external_url=item.get("url"),
                    pdf_url=open_pdf.get("url") if isinstance(open_pdf, dict) else None,
                    reference_count=item.get("referenceCount"),
                    citation_count=item.get("citationCount"),
                    fields_of_study=fields[:12],
                    keywords=fields[:12],
                    is_open_access=bool(open_pdf.get("url")) if isinstance(open_pdf, dict) else False,
                    relevance_score=1.0 - min(index, limit) * 0.03,
                )
            )
        return results


class CrossrefDiscoveryClient:
    def __init__(self, base_url: str = "https://api.crossref.org/works", timeout: int = 6) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, limit: int) -> list[ProviderResult]:
        params = urlencode({"query.bibliographic": query, "rows": str(limit)})
        payload = _get_json(f"{self.base_url}?{params}", "research-learning-agent/0.15", self.timeout)
        items = payload.get("message", {}).get("items", []) if isinstance(payload, dict) else []
        results = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            title = _first(item.get("title"))
            if not title:
                continue
            subjects = [value for value in item.get("subject", []) if isinstance(value, str)]
            results.append(
                ProviderResult(
                    source="crossref",
                    source_id=item.get("DOI"),
                    title=title,
                    authors=_authors_from_crossref(item.get("author", [])),
                    year=_year_from_crossref(item),
                    venue=_first(item.get("container-title")),
                    doi=item.get("DOI"),
                    abstract=_strip_html(item.get("abstract")),
                    external_url=item.get("URL"),
                    reference_count=item.get("reference-count"),
                    citation_count=item.get("is-referenced-by-count"),
                    keywords=subjects[:12],
                    relevance_score=1.0 - min(index, limit) * 0.03,
                )
            )
        return results


class ArxivDiscoveryClient:
    def __init__(self, base_url: str = "https://export.arxiv.org/api/query", timeout: int = 6) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def search(self, query: str, limit: int) -> list[ProviderResult]:
        params = urlencode({"search_query": f"all:{query}", "start": "0", "max_results": str(limit)})
        request = Request(f"{self.base_url}?{params}", headers={"User-Agent": "research-learning-agent/0.15"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError):
            return []

        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        results = []
        for index, entry in enumerate(root.findall("atom:entry", ns)):
            title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
            if not title:
                continue
            authors = [
                _clean_text(author.findtext("atom:name", default="", namespaces=ns)) or ""
                for author in entry.findall("atom:author", ns)
            ]
            links = entry.findall("atom:link", ns)
            pdf_url = None
            external_url = entry.findtext("atom:id", default=None, namespaces=ns)
            for link in links:
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
            categories = [category.attrib.get("term", "") for category in entry.findall("atom:category", ns)]
            published = entry.findtext("atom:published", default="", namespaces=ns)
            results.append(
                ProviderResult(
                    source="arxiv",
                    source_id=external_url,
                    title=title,
                    authors=", ".join(author for author in authors if author) or None,
                    year=_year_from_text(published),
                    venue="arXiv",
                    abstract=_clean_text(entry.findtext("atom:summary", default="", namespaces=ns)),
                    external_url=external_url,
                    pdf_url=pdf_url,
                    fields_of_study=[value for value in categories if value][:12],
                    keywords=[value for value in categories if value][:12],
                    is_open_access=bool(pdf_url),
                    relevance_score=1.0 - min(index, limit) * 0.03,
                )
            )
        return results


class OpenAlexDiscoveryClient:
    def __init__(self, base_url: str = "https://api.openalex.org/works", timeout: int = 6) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def search(self, query: str, limit: int) -> list[ProviderResult]:
        params = urlencode({"search": query, "per-page": str(limit)})
        payload = _get_json(f"{self.base_url}?{params}", "research-learning-agent/0.15", self.timeout)
        results = []
        for index, item in enumerate(payload.get("results", []) if isinstance(payload, dict) else []):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("display_name"))
            if not title:
                continue
            concepts = [concept.get("display_name") for concept in item.get("concepts", []) if isinstance(concept, dict)]
            locations = item.get("primary_location") or {}
            source = locations.get("source") or {}
            open_access = item.get("open_access") or {}
            results.append(
                ProviderResult(
                    source="openalex",
                    source_id=item.get("id"),
                    title=title,
                    authors=_authors_from_openalex(item.get("authorships", [])),
                    year=item.get("publication_year"),
                    venue=source.get("display_name") if isinstance(source, dict) else None,
                    doi=(item.get("doi") or "").replace("https://doi.org/", "") or None,
                    abstract=_abstract_from_openalex(item.get("abstract_inverted_index")),
                    external_url=item.get("id"),
                    pdf_url=open_access.get("oa_url") if isinstance(open_access, dict) else None,
                    citation_count=item.get("cited_by_count"),
                    fields_of_study=[value for value in concepts if value][:12],
                    keywords=[value for value in concepts if value][:12],
                    is_open_access=bool(open_access.get("is_oa")) if isinstance(open_access, dict) else False,
                    relevance_score=1.0 - min(index, limit) * 0.03,
                )
            )
        return results


def _get_json(url: str, user_agent: str, timeout: int) -> dict:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return {}


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    return unescape(re.sub(r"\s+", " ", str(value))).strip() or None


def _first(value) -> str | None:
    if isinstance(value, list) and value:
        return _clean_text(str(value[0]))
    if isinstance(value, str):
        return _clean_text(value)
    return None


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    return _clean_text(re.sub(r"<[^>]+>", " ", value))


def _authors_from_semantic_scholar(authors) -> str | None:
    if not isinstance(authors, list):
        return None
    names = [author.get("name") for author in authors[:20] if isinstance(author, dict) and author.get("name")]
    return ", ".join(names) if names else None


def _authors_from_crossref(authors) -> str | None:
    if not isinstance(authors, list):
        return None
    names = []
    for author in authors[:20]:
        if not isinstance(author, dict):
            continue
        name = " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part).strip()
        if name:
            names.append(name)
    return ", ".join(names) if names else None


def _authors_from_openalex(authorships) -> str | None:
    if not isinstance(authorships, list):
        return None
    names = []
    for authorship in authorships[:20]:
        author = authorship.get("author") if isinstance(authorship, dict) else None
        if isinstance(author, dict) and author.get("display_name"):
            names.append(author["display_name"])
    return ", ".join(names) if names else None


def _year_from_crossref(message: dict) -> int | None:
    for key in ["published-print", "published-online", "published", "issued", "created"]:
        date = message.get(key)
        if not isinstance(date, dict):
            continue
        date_parts = date.get("date-parts")
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            year = date_parts[0][0]
            if isinstance(year, int):
                return year
    return None


def _year_from_text(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19[8-9]\d|20[0-4]\d)\b", value)
    return int(match.group(1)) if match else None


def _abstract_from_openalex(index) -> str | None:
    if not isinstance(index, dict):
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words.append((position, word))
    return _clean_text(" ".join(word for _, word in sorted(words))) if words else None
