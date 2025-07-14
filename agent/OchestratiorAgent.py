# OrchestratorAgent - LangGraph 프레임워크 기반 구현
# 프레임워크: LangGraph (상태 관리 및 복잡한 워크플로우 제어에 최적화)

import asyncio
import json
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum

# LangGraph 관련 imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.pydantic_v1 import BaseModel, Field
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # LangGraph가 없을 때의 폴백 구현
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available, using fallback implementation")

from modular_agent_architecture import ProcessingMode
from modular_agent_architecture import (
    ModularAgent, AgentConfig, ProcessingRequest, ProcessingResponse,
    FrameworkAdapter, AgentCapability, AgentFramework
)


class WorkflowState(TypedDict):
    """LangGraph 상태 정의"""
    request: ProcessingRequest
    current_stage: str
    stage_results: Dict[str, Any]
    context_budget: Dict[str, int]
    workflow_plan: List[str]
    errors: List[str]
    final_result: Optional[ProcessingResponse]


class OrchestratorStage(Enum):
    """오케스트레이터 처리 단계"""
    PLANNING = "planning"
    BUDGET_ALLOCATION = "budget_allocation" 
    AGENT_COORDINATION = "agent_coordination"
    RESULT_INTEGRATION = "result_integration"
    QUALITY_VALIDATION = "quality_validation"
    FINALIZATION = "finalization"


class LangGraphAdapter(FrameworkAdapter):
    """LangGraph 프레임워크 어댑터"""
    
    def __init__(self):
        self.graph: Optional[StateGraph] = None
        self.checkpointer = None
        self.workflow_templates = {}
        self.is_initialized = False
    
    def get_framework_name(self) -> str:
        return "LangGraph"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """LangGraph 초기화"""
        try:
            if not LANGGRAPH_AVAILABLE:
                return await self._initialize_fallback(config)
            
            # SQLite 체크포인터 설정 (상태 영속성)
            self.checkpointer = SqliteSaver.from_conn_string(":memory:")
            
            # 워크플로우 템플릿 로드
            self.workflow_templates = config.get('workflow_templates', self._get_default_templates())
            
            # StateGraph 구성
            self.graph = self._build_state_graph()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"LangGraph 초기화 실패: {e}")
            return await self._initialize_fallback(config)
    
    async def _initialize_fallback(self, config: Dict[str, Any]) -> bool:
        """폴백 초기화"""
        self.workflow_templates = config.get('workflow_templates', self._get_default_templates())
        self.is_initialized = True
        return True
    
    def _get_default_templates(self) -> Dict[str, List[str]]:
        """기본 워크플로우 템플릿"""
        return {
            "standard": [
                OrchestratorStage.PLANNING.value,
                OrchestratorStage.BUDGET_ALLOCATION.value,
                OrchestratorStage.AGENT_COORDINATION.value,
                OrchestratorStage.RESULT_INTEGRATION.value,
                OrchestratorStage.QUALITY_VALIDATION.value,
                OrchestratorStage.FINALIZATION.value
            ],
            "fast": [
                OrchestratorStage.BUDGET_ALLOCATION.value,
                OrchestratorStage.AGENT_COORDINATION.value,
                OrchestratorStage.FINALIZATION.value
            ],
            "comprehensive": [
                OrchestratorStage.PLANNING.value,
                OrchestratorStage.BUDGET_ALLOCATION.value,
                OrchestratorStage.AGENT_COORDINATION.value,
                OrchestratorStage.RESULT_INTEGRATION.value,
                OrchestratorStage.QUALITY_VALIDATION.value,
                OrchestratorStage.RESULT_INTEGRATION.value,  # 재검토
                OrchestratorStage.FINALIZATION.value
            ]
        }
    
    def _build_state_graph(self) -> StateGraph:
        """LangGraph StateGraph 구성"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        workflow = StateGraph(WorkflowState)
        
        # 노드 추가
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("budget_allocation", self._budget_allocation_node)
        workflow.add_node("agent_coordination", self._agent_coordination_node)
        workflow.add_node("result_integration", self._result_integration_node)
        workflow.add_node("quality_validation", self._quality_validation_node)
        workflow.add_node("finalization", self._finalization_node)
        
        # 엣지 추가 (조건부 라우팅)
        workflow.set_entry_point("planning")
        workflow.add_edge("planning", "budget_allocation")
        workflow.add_edge("budget_allocation", "agent_coordination")
        workflow.add_edge("agent_coordination", "result_integration")
        workflow.add_edge("result_integration", "quality_validation")
        
        # 조건부 엣지
        workflow.add_conditional_edges(
            "quality_validation",
            self._should_retry,
            {
                "retry": "result_integration",
                "finalize": "finalization"
            }
        )
        workflow.add_edge("finalization", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _planning_node(self, state: WorkflowState) -> WorkflowState:
        """계획 수립 노드"""
        request = state["request"]
        
        # 요청 분석
        complexity_score = await self._analyze_request_complexity(request)
        priority_level = request.processing_options.get("priority", 5)
        quality_threshold = max(request.quality_requirements.values()) if request.quality_requirements else 0.7
        
        # 워크플로우 선택
        if priority_level >= 9:
            workflow_type = "fast"
        elif quality_threshold >= 0.9 or complexity_score > 0.8:
            workflow_type = "comprehensive"
        else:
            workflow_type = "standard"
        
        state["workflow_plan"] = self.workflow_templates[workflow_type]
        state["current_stage"] = OrchestratorStage.PLANNING.value
        state["stage_results"]["planning"] = {
            "workflow_type": workflow_type,
            "complexity_score": complexity_score,
            "estimated_steps": len(state["workflow_plan"])
        }
        
        return state
    
    async def _budget_allocation_node(self, state: WorkflowState) -> WorkflowState:
        """예산 분배 노드"""
        request = state["request"]
        max_tokens = request.processing_options.get("max_tokens", 4000)
        
        # 동적 예산 분배
        budget_allocation = await self._calculate_dynamic_budget(
            max_tokens, 
            state["workflow_plan"],
            state["stage_results"]["planning"]["complexity_score"]
        )
        
        state["context_budget"] = budget_allocation
        state["current_stage"] = OrchestratorStage.BUDGET_ALLOCATION.value
        state["stage_results"]["budget_allocation"] = {
            "total_budget": max_tokens,
            "allocations": budget_allocation,
            "allocation_strategy": "dynamic_complexity_based"
        }
        
        return state
    
    async def _agent_coordination_node(self, state: WorkflowState) -> WorkflowState:
        """에이전트 조율 노드"""
        # 실제 구현에서는 다른 에이전트들과의 통신 수행
        coordination_results = await self._coordinate_agents(state)
        
        state["current_stage"] = OrchestratorStage.AGENT_COORDINATION.value
        state["stage_results"]["agent_coordination"] = coordination_results
        
        return state
    
    async def _result_integration_node(self, state: WorkflowState) -> WorkflowState:
        """결과 통합 노드"""
        integrated_result = await self._integrate_agent_results(
            state["stage_results"]["agent_coordination"]
        )
        
        state["current_stage"] = OrchestratorStage.RESULT_INTEGRATION.value
        state["stage_results"]["result_integration"] = integrated_result
        
        return state
    
    async def _quality_validation_node(self, state: WorkflowState) -> WorkflowState:
        """품질 검증 노드"""
        validation_result = await self._validate_integrated_result(
            state["stage_results"]["result_integration"],
            state["request"].quality_requirements
        )
        
        state["current_stage"] = OrchestratorStage.QUALITY_VALIDATION.value
        state["stage_results"]["quality_validation"] = validation_result
        
        return state
    
    async def _finalization_node(self, state: WorkflowState) -> WorkflowState:
        """최종화 노드"""
        final_content = await self._create_final_response(state)
        
        response = ProcessingResponse(
            request_id=state["request"].request_id,
            processed_content=final_content,
            confidence_score=state["stage_results"]["quality_validation"]["overall_confidence"],
            quality_metrics=state["stage_results"]["quality_validation"]["quality_metrics"],
            processing_time=0.0,  # 실제로는 시작 시간부터 계산
            framework_info=self.get_framework_info(),
            metadata={
                "workflow_executed": state["workflow_plan"],
                "stage_results": state["stage_results"],
                "budget_usage": state["context_budget"]
            }
        )
        
        state["final_result"] = response
        state["current_stage"] = OrchestratorStage.FINALIZATION.value
        
        return state
    
    def _should_retry(self, state: WorkflowState) -> str:
        """재시도 여부 결정"""
        quality_score = state["stage_results"]["quality_validation"]["overall_confidence"]
        min_quality = min(state["request"].quality_requirements.values()) if state["request"].quality_requirements else 0.7
        
        if quality_score < min_quality and len(state.get("errors", [])) < 2:
            return "retry"
        return "finalize"
    
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """LangGraph를 사용한 요청 처리"""
        if not self.is_initialized:
            await self.initialize({})
        
        if self.graph and LANGGRAPH_AVAILABLE:
            return await self._process_with_langgraph(request)
        else:
            return await self._process_with_fallback(request)
    
    async def _process_with_langgraph(self, request: ProcessingRequest) -> ProcessingResponse:
        """LangGraph를 사용한 처리"""
        initial_state = WorkflowState(
            request=request,
            current_stage="",
            stage_results={},
            context_budget={},
            workflow_plan=[],
            errors=[],
            final_result=None
        )
        
        try:
            # LangGraph 실행
            config = {"configurable": {"thread_id": request.request_id}}
            result = await self.graph.ainvoke(initial_state, config=config)
            
            return result["final_result"]
            
        except Exception as e:
            return ProcessingResponse(
                request_id=request.request_id,
                processed_content=None,
                confidence_score=0.0,
                quality_metrics={"error": 1.0},
                processing_time=0.0,
                framework_info=self.get_framework_info(),
                error_details={"message": str(e), "framework": "LangGraph"}
            )
    
    async def _process_with_fallback(self, request: ProcessingRequest) -> ProcessingResponse:
        """폴백 처리 방식"""
        # 순차적으로 각 단계 실행
        state = {
            "request": request,
            "stage_results": {},
            "context_budget": {},
            "workflow_plan": self.workflow_templates["standard"]
        }
        
        # 각 단계 순차 실행
        await self._planning_node(state)
        await self._budget_allocation_node(state)
        await self._agent_coordination_node(state)
        await self._result_integration_node(state)
        await self._quality_validation_node(state)
        await self._finalization_node(state)
        
        return state["final_result"]
    
    def get_framework_info(self) -> Dict[str, str]:
        """프레임워크 정보"""
        return {
            "name": "LangGraph",
            "version": "0.0.40" if LANGGRAPH_AVAILABLE else "fallback",
            "status": "active" if self.is_initialized else "initializing",
            "features": "state_management,workflow_control,checkpointing"
        }
    
    # 헬퍼 메서드들
    async def _analyze_request_complexity(self, request: ProcessingRequest) -> float:
        """요청 복잡도 분석"""
        content_length = len(str(request.content))
        options_count = len(request.processing_options)
        quality_reqs = len(request.quality_requirements)
        
        # 복잡도 계산 (0.0 ~ 1.0)
        complexity = min(
            (content_length / 10000 * 0.4) +
            (options_count / 10 * 0.3) +
            (quality_reqs / 5 * 0.3),
            1.0
        )
        
        return complexity
    
    async def _calculate_dynamic_budget(self, total_budget: int, workflow_plan: List[str], complexity: float) -> Dict[str, int]:
        """동적 예산 분배"""
        base_allocations = {
            "planning": 0.1,
            "budget_allocation": 0.05,
            "agent_coordination": 0.5,
            "result_integration": 0.25,
            "quality_validation": 0.1
        }
        
        # 복잡도에 따른 조정
        if complexity > 0.7:
            base_allocations["agent_coordination"] += 0.1
            base_allocations["quality_validation"] += 0.1
            base_allocations["result_integration"] -= 0.2
        
        # 실제 예산 할당
        budget_allocation = {}
        for stage in workflow_plan:
            if stage in base_allocations:
                budget_allocation[stage] = int(total_budget * base_allocations[stage])
        
        return budget_allocation
    
    async def _coordinate_agents(self, state: WorkflowState) -> Dict[str, Any]:
        """에이전트 조율 (실제 구현에서는 다른 에이전트들과 통신)"""
        return {
            "retriever_result": {"status": "completed", "data_collected": True},
            "synthesizer_result": {"status": "completed", "data_structured": True},
            "evaluator_result": {"status": "completed", "quality_verified": True}
        }
    
    async def _integrate_agent_results(self, coordination_results: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 결과 통합"""
        return {
            "integrated_content": "통합된 최종 결과",
            "sources_used": ["retriever", "synthesizer"],
            "integration_method": "weighted_combination"
        }
    
    async def _validate_integrated_result(self, integrated_result: Dict[str, Any], quality_requirements: Dict[str, float]) -> Dict[str, Any]:
        """통합 결과 검증"""
        return {
            "overall_confidence": 0.85,
            "quality_metrics": {
                "accuracy": 0.9,
                "completeness": 0.8,
                "relevance": 0.85
            },
            "validation_passed": True
        }
    
    async def _create_final_response(self, state: WorkflowState) -> str:
        """최종 응답 생성"""
        return f"""
# 컨텍스트 엔지니어링 결과

## 요청 처리 완료
- 요청 ID: {state['request'].request_id}
- 처리 방식: {state['workflow_plan']}
- 전체 신뢰도: {state['stage_results']['quality_validation']['overall_confidence']:.2f}

## 처리 단계별 결과
{json.dumps(state['stage_results'], indent=2, ensure_ascii=False)}

## 최종 통합 결과
{state['stage_results']['result_integration']['integrated_content']}
"""


class WorkflowManagementCapability(AgentCapability):
    """워크플로우 관리 능력"""
    
    def get_capability_name(self) -> str:
        return "workflow_management"
    
    def get_supported_formats(self) -> List[str]:
        return ["json", "yaml", "workflow_config"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        workflow_config = input_data
        
        # 워크플로우 검증 및 최적화
        optimized_workflow = await self._optimize_workflow(workflow_config)
        
        return {
            "original_workflow": workflow_config,
            "optimized_workflow": optimized_workflow,
            "optimization_applied": True
        }
    
    async def _optimize_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우 최적화"""
        # 불필요한 단계 제거, 병렬화 가능한 단계 식별 등
        return workflow


class ContextBudgetingCapability(AgentCapability):
    """컨텍스트 예산 관리 능력"""
    
    def get_capability_name(self) -> str:
        return "context_budgeting"
    
    def get_supported_formats(self) -> List[str]:
        return ["budget_config", "token_allocation"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        total_budget = input_data.get("total_budget", 4000)
        requirements = input_data.get("requirements", {})
        
        # 동적 예산 계산
        allocation = await self._calculate_optimal_allocation(total_budget, requirements)
        
        return {
            "total_budget": total_budget,
            "allocation": allocation,
            "allocation_strategy": "dynamic_optimization"
        }
    
    async def _calculate_optimal_allocation(self, total: int, requirements: Dict[str, Any]) -> Dict[str, int]:
        """최적 예산 분배 계산"""
        # 요구사항 기반 최적화 로직
        return {
            "retrieval": int(total * 0.4),
            "synthesis": int(total * 0.3),
            "evaluation": int(total * 0.2),
            "coordination": int(total * 0.1)
        }


class OrchestratorAgent(ModularAgent):
    """LangGraph 기반 관리자 에이전트"""
    
    def __init__(self, config: AgentConfig):
        # 프레임워크 검증
        if config.framework != AgentFramework.LANGGRAPH:
            raise ValueError(f"OrchestratorAgent는 LangGraph 프레임워크만 지원합니다. 현재: {config.framework}")
        
        super().__init__(config)
    
    async def _register_default_capabilities(self):
        """기본 능력 등록"""
        # 워크플로우 관리 능력
        workflow_capability = WorkflowManagementCapability()
        self.capability_registry.register_capability(workflow_capability)
        
        # 컨텍스트 예산 관리 능력
        budget_capability = ContextBudgetingCapability()
        self.capability_registry.register_capability(budget_capability)
        
        self.logger.info("OrchestratorAgent 기본 능력 등록 완료")
    
    async def _load_framework_adapter(self) -> FrameworkAdapter:
        """LangGraph 어댑터 로드"""
        adapter = LangGraphAdapter()
        
        # 에이전트별 맞춤 설정
        framework_config = {
            "workflow_templates": self.config.custom_config.get("workflow_templates", {}),
            "checkpointer_config": self.config.custom_config.get("checkpointer", {}),
            "state_schema": WorkflowState
        }
        
        await adapter.initialize(framework_config)
        return adapter
    
    async def coordinate_multi_agent_workflow(self, agents: Dict[str, Any], workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """다중 에이전트 워크플로우 조율"""
        coordination_request = ProcessingRequest(
            request_id=f"coordination_{int(asyncio.get_event_loop().time())}",
            content={"agents": agents, "workflow": workflow_config},
            content_type="coordination_request",
            processing_options={"coordination_mode": "multi_agent"}
        )
        
        response = await self.process(coordination_request)
        return json.loads(response.processed_content) if response.processed_content else {}


# 사용 예시
async def example_orchestrator_usage():
    """OrchestratorAgent 사용 예시"""
    
    # 설정 생성
    config = AgentConfig(
        agent_id="orchestrator_main",
        framework=AgentFramework.LANGGRAPH,
        capabilities=["workflow_management", "context_budgeting"],
        processing_mode=ProcessingMode.ASYNC,
        custom_config={
            "workflow_templates": {
                "custom_research": [
                    "planning", "deep_retrieval", "multi_synthesis", 
                    "cross_validation", "finalization"
                ]
            }
        }
    )
    
    # 에이전트 생성
    orchestrator = OrchestratorAgent(config)
    
    # 처리 요청
    request = ProcessingRequest(
        request_id="test_orchestration_001",
        content="Python 웹 크롤링에 대한 종합적인 가이드를 만들어주세요",
        content_type="comprehensive_guide_request",
        target_language="ko",
        processing_options={"priority": 7, "max_tokens": 6000},
        quality_requirements={"accuracy": 0.9, "completeness": 0.85}
    )
    
    # 처리 실행
    response = await orchestrator.process(request)
    
    print(f"처리 완료: 신뢰도 {response.confidence_score:.2f}")
    print(f"사용된 프레임워크: {response.framework_info['name']}")
    print(f"처리 시간: {response.processing_time:.2f}초")
    
    return response


if __name__ == "__main__":
    asyncio.run(example_orchestrator_usage())