from typing import Optional

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

class RelevanceEvaluatorTool(BaseTool):
    """관련성 평가 도구 (LangChain)"""
    name = "relevance_evaluator"
    description = "검색된 콘텐츠와 쿼리 간의 관련성을 평가합니다"

    def _run(self, content: str, query: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """관련성 평가"""
        # 간단한 키워드 기반 평가
        query_words = set(query.lower().split()) if query else set()
        content_words = set(content.lower().split())
        if query_words:
            overlap = len(query_words.intersection(content_words))
            relevance_score = min(overlap / len(query_words), 1.0)
        else:
            relevance_score = 0.8

        return f"""
# 관련성 평가 결과

**평가 점수**: {relevance_score:.2f} / 1.0
**권장사항**: {'포함' if relevance_score > 0.7 else '검토 필요'}
"""

    async def _arun(self, content: str, query: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """비동기 실행 (동기 메서드 호출)"""
        return self._run(content, query, run_manager)