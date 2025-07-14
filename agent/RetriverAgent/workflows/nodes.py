import json
from typing import Dict, Any

from langchain.schema import SystemMessage, HumanMessage

from ..models.state import AgentState
from ..models.data_models import RetrievalTask
from ..utils.exceptions import handle_langchain_exceptions

class WorkflowNodes:
    """워크플로우 노드 컬렉션"""

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents

    @handle_langchain_exceptions(fallback_value=None)
    async def web_search_node(self, state: AgentState) -> AgentState:
        """웹 검색 노드 - 메시지 이력 관리 개선"""
        if "web_specialist" in self.agents:
            result = await self.agents["web_specialist"].ainvoke({"input": state["query"]})
            state["results"]["web_search"] = result["output"]

            # 메시지 이력에 결과 추가
            state["messages"].append(
                SystemMessage(content=f"웹 검색 완료: {result['output'][:100]}...")
            )

            # 처리 이력 업데이트
            state["processing_history"].append("web_search_completed")

        state["current_step"] = "api_search"
        return state

    @handle_langchain_exceptions(fallback_value=None)
    async def api_search_node(self, state: AgentState) -> AgentState:
        """API 검색 노드"""
        if "api_specialist" in self.agents:
            result = await self.agents["api_specialist"].ainvoke({"input": state["query"]})
            state["results"]["api_search"] = result["output"]

            state["messages"].append(
                SystemMessage(content=f"API 검색 완료: {result['output'][:100]}...")
            )

            state["processing_history"].append("api_search_completed")

        state["current_step"] = "multimedia_process"
        return state

    @handle_langchain_exceptions(fallback_value=None)
    async def multimedia_node(self, state: AgentState) -> AgentState:
        """멀티미디어 처리 노드"""
        if "multimedia_specialist" in self.agents:
            result = await self.agents["multimedia_specialist"].ainvoke({"input": state["query"]})
            state["results"]["multimedia"] = result["output"]

            state["messages"].append(
                SystemMessage(content=f"멀티미디어 분석 완료: {result['output'][:100]}...")
            )

            state["processing_history"].append("multimedia_completed")

        state["current_step"] = "quality_evaluation"
        return state

    @handle_langchain_exceptions(fallback_value=None)
    async def quality_node(self, state: AgentState) -> AgentState:
        """품질 평가 노드"""
        if "quality_specialist" in self.agents:
            all_results = json.dumps(state["results"], ensure_ascii=False)
            result = await self.agents["quality_specialist"].ainvoke({"input": all_results, "query": state["query"]})
            state["results"]["quality_evaluation"] = result["output"]

            state["messages"].append(
                SystemMessage(content=f"품질 평가 완료: {result['output'][:100]}...")
            )

            state["processing_history"].append("quality_evaluation_completed")

        state["current_step"] = "complete"
        return state

    def create_initial_state(self, query: str, task: RetrievalTask, metadata: Dict[str, Any]) -> AgentState:
        """워크플로우 초기 상태 생성"""
        return AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            results={},
            current_step="web_search",
            metadata=metadata,
            task_info=task.to_dict(),
            processing_history=["initialized"]
        )