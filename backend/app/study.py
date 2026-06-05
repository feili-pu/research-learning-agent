from .answerer import Answerer
from .rag import RagStore
from .schemas import SourceChunk, StudyRequest, StudyResponse


class StudyService:
    def __init__(self, store: RagStore, answerer: Answerer) -> None:
        self.store = store
        self.answerer = answerer

    def summary(self, request: StudyRequest) -> StudyResponse:
        prompt = self._build_prompt(
            task="summary",
            topic=request.topic,
            focus=request.focus,
            instruction=(
                "请基于检索到的资料，生成一份研究生可读的中文总览。"
                "包括：研究主题、核心问题、主要方法、关键结论、适合继续追问的方向。"
            ),
        )
        return self._run("summary", request, prompt)

    def key_points(self, request: StudyRequest) -> StudyResponse:
        prompt = self._build_prompt(
            task="key_points",
            topic=request.topic,
            focus=request.focus,
            instruction=(
                "请基于检索到的资料，提炼学习关键点。"
                "按概念、方法、数据/实验、结论、局限和可写进笔记的句子来组织。"
            ),
        )
        return self._run("key_points", request, prompt)

    def reading_plan(self, request: StudyRequest) -> StudyResponse:
        prompt = self._build_prompt(
            task="reading_plan",
            topic=request.topic,
            focus=request.focus,
            instruction=(
                "请基于检索到的资料，生成一个循序渐进的阅读计划。"
                "包含阅读顺序、每一步目标、需要记录的问题、最后的复盘任务。"
            ),
        )
        return self._run("reading_plan", request, prompt)

    def _run(self, task: str, request: StudyRequest, prompt: str) -> StudyResponse:
        results = self.store.search(prompt, request.top_k, request.section_filter)
        answer = self.answerer.answer(prompt, results)
        return StudyResponse(
            task=task,
            topic=request.topic,
            retrieval_mode=self.store.active_retrieval_mode,
            answer_mode=answer.answer_mode,
            model=answer.model,
            answer=answer.answer,
            sources=[
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
            ],
        )

    def _build_prompt(
        self,
        task: str,
        topic: str,
        focus: str | None,
        instruction: str,
    ) -> str:
        focus_text = f"\n关注重点：{focus}" if focus else ""
        return (
            f"学习任务：{task}\n"
            f"主题：{topic}"
            f"{focus_text}\n"
            f"要求：{instruction}\n"
            "请只依据资料来源回答，并尽量引用来源编号。"
        )
