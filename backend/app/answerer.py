from dataclasses import dataclass
import os

from .rag import SearchResult

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


@dataclass
class AnswerResult:
    answer: str
    answer_mode: str
    model: str | None


class Answerer:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = os.getenv("RLA_LLM_MODEL", model)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = self._read_base_url()
        self.wire_api = os.getenv("RLA_OPENAI_WIRE_API", "responses").strip().lower()
        self.client = self._build_client()

    def answer(self, question: str, results: list[SearchResult]) -> AnswerResult:
        if not results:
            return AnswerResult(
                answer=(
                    "I could not find relevant content in the uploaded documents yet. "
                    "Try uploading a PDF that contains this topic, or ask with more specific keywords."
                ),
                answer_mode="no_sources",
                model=None,
            )

        if self.client is None:
            return AnswerResult(
                answer=self._retrieval_only_answer(
                    question,
                    results,
                    "V3 found relevant sources, but no LLM API key is configured yet.",
                ),
                answer_mode="retrieval_only",
                model=None,
            )

        prompt = self._build_prompt(question, results)
        try:
            answer = self._call_llm(prompt)
            return AnswerResult(
                answer=answer,
                answer_mode="llm",
                model=self.model,
            )
        except Exception as exc:
            fallback = self._retrieval_only_answer(
                question,
                results,
                "V3 found relevant sources, but LLM generation failed.",
            )
            return AnswerResult(
                answer=(
                    "LLM generation failed, so V3 fell back to retrieval-only mode. "
                    f"Error type: {type(exc).__name__}\n\n{fallback}"
                ),
                answer_mode="llm_error_fallback",
                model=None,
            )

    def _retrieval_only_answer(
        self,
        question: str,
        results: list[SearchResult],
        reason: str,
    ) -> str:
        excerpts = []
        for index, result in enumerate(results, start=1):
            text = self._shorten(result.chunk.text, max_chars=500)
            excerpts.append(f"[{index}] Page {result.chunk.page}: {text}")

        joined = "\n".join(excerpts)
        return (
            f"{reason} Here are the retrieved chunks to ground your answer.\n\n"
            f"Question: {question}\n\n"
            f"Sources:\n{joined}"
        )

    def _build_prompt(self, question: str, results: list[SearchResult]) -> str:
        sources = []
        for index, result in enumerate(results, start=1):
            sources.append(
                "\n".join(
                    [
                        f"[{index}]",
                        f"filename: {result.chunk.filename}",
                        f"page: {result.chunk.page}",
                        f"chunk_id: {result.chunk.chunk_id}",
                        f"text: {result.chunk.text}",
                    ]
                )
            )

        joined_sources = "\n\n".join(sources)
        return (
            f"Answer this question:\n{question}\n\n"
            f"Sources:\n{joined_sources}"
        )

    def _shorten(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _read_base_url(self) -> str | None:
        base_url = os.getenv("RLA_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not base_url:
            return None
        return base_url.rstrip("/")

    def _build_client(self):
        if OpenAI is None or not self.api_key:
            return None

        if self.base_url:
            return OpenAI(api_key=self.api_key, base_url=self.base_url)

        return OpenAI(api_key=self.api_key)

    def _call_llm(self, prompt: str) -> str:
        if self.wire_api == "chat":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research learning assistant. Answer using only the provided sources. "
                            "If the sources are insufficient, say what is missing. Cite sources with bracket numbers like [1]."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a research learning assistant. Answer using only the provided sources. "
                        "If the sources are insufficient, say what is missing. Cite sources with bracket numbers like [1]."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text
