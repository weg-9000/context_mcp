from typing import Dict, List, Any
from typing_extensions import TypedDict

from langchain.schema import BaseMessage

class AgentState(TypedDict):
    """LangGraph 상태 정의 - 메시지 이력 활용 개선"""
    messages: List[BaseMessage]
    query: str
    results: Dict[str, Any]
    current_step: str
    metadata: Dict[str, Any]
    task_info: Dict[str, Any]  # RetrievalTask 정보
    processing_history: List[str]  # 처리 이력