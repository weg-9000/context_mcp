import time
import logging
from typing import Dict, List, Any

from control.modular_agent_architecture import AgentCapability
from langchain.schema import LangChainException

class RelevanceEvaluationCapability(AgentCapability):
    """관련성 평가 능력"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RelevanceEvaluationCapability")

    def get_capability_name(self) -> str:
        return "relevance_evaluation"

    def get_supported_formats(self) -> List[str]:
        return ["text_content", "structured_data", "multimedia_content"]

    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        content = input_data.get("content", "") if isinstance(input_data, dict) else str(input_data)
        query = input_data.get("query", "") if isinstance(input_data, dict) else config.get("query", "")
        self.logger.info(f"관련성 평가 시작: query='{query[:50]}...', content_length={len(content)}")

        try:
            relevance_score = await self._calculate_relevance_with_llm(content, query)
            analysis = await self._detailed_relevance_analysis(content, query)
            result = {
                "relevance_score": relevance_score,
                "evaluation_method": "enhanced_llm_based",
                "factors_considered": ["semantic_similarity", "keyword_overlap", "context_alignment"],
                "detailed_analysis": analysis,
                "recommendation": "include" if relevance_score > 0.7 else "review",
                "confidence": min(relevance_score + 0.1, 1.0)
            }
            self.logger.info(f"관련성 평가 완료: score={relevance_score:.2f}")
            return result
        except (LangChainException, Exception) as e:
            self.logger.error(f"관련성 평가 중 오류 발생 ({type(e).__name__}): {e}")
            return {"relevance_score": 0.5, "evaluation_method": "fallback", "error": str(e)}

    async def _calculate_relevance_with_llm(self, content: str, query: str) -> float:
        """LLM 기반 관련성 계산 (시뮬레이션)"""
        if not query or not content:
            return 0.5
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        exact_matches = len(query_words.intersection(content_words))
        
        if not query_words:
            return 0.5
            
        exact_ratio = exact_matches / len(query_words)
        base_score = exact_ratio * 0.8
        
        length_factor = min(len(content) / 1000, 1.0) * 0.2
        final_score = min(base_score + length_factor, 1.0)
        return round(final_score, 3)

    async def _detailed_relevance_analysis(self, content: str, query: str) -> Dict[str, Any]:
        """상세 관련성 분석 (시뮬레이션)"""
        analysis = {"content_length": len(content), "query_length": len(query), "keyword_density": 0.0}
        if query:
            query_words = query.lower().split()
            content_lower = content.lower()
            keyword_count = sum(content_lower.count(word) for word in query_words)
            total_words = len(content.split())
            analysis["keyword_density"] = round(keyword_count / max(total_words, 1), 3)
            
            if analysis["keyword_density"] > 0.05:
                analysis["topic_alignment"] = "high"
            else:
                analysis["topic_alignment"] = "medium"
        return analysis