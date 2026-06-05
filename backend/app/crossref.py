import json
import re
from dataclasses import dataclass, field
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass
class CrossrefWork:
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    publisher: str | None = None
    external_url: str | None = None
    reference_count: int | None = None
    keywords: list[str] = field(default_factory=list)


class CrossrefClient:
    def __init__(self, base_url: str = "https://api.crossref.org/works", timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_by_doi(self, doi: str) -> CrossrefWork | None:
        clean_doi = doi.strip()
        if not clean_doi:
            return None

        url = f"{self.base_url}/{quote(clean_doi, safe='')}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "research-learning-agent/0.9 (mailto:research-learning-agent@example.local)",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return None

        message = payload.get("message", {})
        if not isinstance(message, dict):
            return None

        return CrossrefWork(
            title=self._first(message.get("title")),
            authors=self._authors(message.get("author", [])),
            year=self._year(message),
            venue=self._first(message.get("container-title")),
            doi=message.get("DOI") or clean_doi,
            abstract=self._abstract(message.get("abstract")),
            publisher=message.get("publisher"),
            external_url=message.get("URL"),
            reference_count=message.get("reference-count") or message.get("is-referenced-by-count"),
            keywords=[item for item in message.get("subject", []) if isinstance(item, str)][:12],
        )

    def _first(self, value) -> str | None:
        if isinstance(value, list) and value:
            return unescape(str(value[0])).strip() or None
        if isinstance(value, str):
            return unescape(value).strip() or None
        return None

    def _authors(self, authors) -> str | None:
        if not isinstance(authors, list):
            return None

        names = []
        for author in authors[:20]:
            if not isinstance(author, dict):
                continue
            given = author.get("given", "")
            family = author.get("family", "")
            name = " ".join(part for part in [given, family] if part).strip()
            if name:
                names.append(name)

        return ", ".join(names) if names else None

    def _year(self, message: dict) -> int | None:
        for key in ["published-print", "published-online", "published", "issued", "created"]:
            date = message.get(key)
            if not isinstance(date, dict):
                continue
            date_parts = date.get("date-parts")
            if (
                isinstance(date_parts, list)
                and date_parts
                and isinstance(date_parts[0], list)
                and date_parts[0]
            ):
                year = date_parts[0][0]
                if isinstance(year, int):
                    return year
        return None

    def _abstract(self, abstract: str | None) -> str | None:
        if not abstract:
            return None
        text = re.sub(r"<[^>]+>", " ", abstract)
        text = unescape(re.sub(r"\s+", " ", text)).strip()
        return text or None
