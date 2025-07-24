import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from ..models.state import AgentState
from ..workflows.nodes import WorkflowNodes

class WorkflowBuilder:
    """LangGraph 워크플로우를 구성하고 컴파일하는 클래스"""

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.nodes = WorkflowNodes(agents)
        self.logger = logging.getLogger(f"{__name__}.WorkflowBuilder")

    def build(self) -> StateGraph:
        """
        에이전트 워크플로우를 구성합니다.

        Returns:
            StateGraph: 컴파일된 LangGraph 애플리케이션
        """
        if not self.agents:
            self.logger.warning("에이전트가 없어 워크플로우를 구성할 수 없습니다.")
            return None

        try:
            workflow = StateGraph(AgentState)

            # 워크플로우에 노드 추가
            workflow.add_node("web_search", self.nodes.web_search_node)
            workflow.add_node("api_search", self.nodes.api_search_node)
            workflow.add_node("multimedia_process", self.nodes.multimedia_node)
            workflow.add_node("quality_evaluation", self.nodes.quality_node)

            # 엣지 정의 (순차 실행)
            # 웹 검색 -> API 검색 -> 멀티미디어 처리 -> 품질 평가 순으로 진행
            workflow.add_edge("web_search", "api_search")
            workflow.add_edge("api_search", "multimedia_process")
            workflow.add_edge("multimedia_process", "quality_evaluation")
            workflow.add_edge("quality_evaluation", END) # 품질 평가 후 종료

            # 시작점 설정
            workflow.set_entry_point("web_search")

            # 그래프 컴파일
            compiled_workflow = workflow.compile()
            self.logger.info("LangGraph 워크플로우 구성 및 컴파일 완료")
            return compiled_workflow

        except Exception as e:
            self.logger.error(f"워크플로우 구성 실패: {e}")
            return None