import asyncio
import time
import json
import importlib
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from agent_isolate.agent_isolation_plugins import PluginManager
from agent_isolate.communication_isolator import UniversalCommunicationIsolator, DataTransferRequest
from agent_isolate.session_manager import UniversalSessionManager, SessionConfig
from agent_isolate.isolation_manager import UniversalIsolationManager
from agent_isolate.config_manager import ConfigManager
from agent_logger.hybridlogging import HybridLogger
from agent_logger.logging_manager import LoggingManager

class AgentFramework(Enum):
    """각 에이전트별 최적화된 프레임워크"""
    LANGGRAPH = "langgraph"      # 관리자 에이전트 - 상태 관리 및 워크플로우
    CREWAI = "crewai"           # 정보 수집 에이전트 - 역할 기반 협업
    SEMANTIC_KERNEL = "semantic_kernel"  # 가공 에이전트 - 플러그인 아키텍처
    LANGCHAIN = "langchain"     # 평가 에이전트 - 평가 도구 생태계


class ProcessingMode(Enum):
    """처리 모드"""
    SYNC = "synchronous"
    ASYNC = "asynchronous" 
    STREAMING = "streaming"
    BATCH = "batch"


@runtime_checkable
class AgentCapability(Protocol):
    """에이전트 능력 프로토콜"""
    
    def get_capability_name(self) -> str:
        """능력 이름 반환"""
        ...
    
    def get_supported_formats(self) -> List[str]:
        """지원 형식 목록 반환"""
        ...
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        """능력 실행"""
        ...


@dataclass
class AgentConfig:
    """에이전트 설정 클래스"""
    agent_id: str
    framework: AgentFramework
    capabilities: List[str] = field(default_factory=list)
    processing_mode: ProcessingMode = ProcessingMode.ASYNC
    max_concurrency: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3
    language_config: Dict[str, Any] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingRequest:
    """범용 처리 요청"""
    request_id: str
    content: Any
    content_type: str
    target_language: str = "auto"
    processing_options: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResponse:
    """범용 처리 응답"""
    request_id: str
    processed_content: Any
    confidence_score: float
    quality_metrics: Dict[str, float]
    processing_time: float
    framework_info: Dict[str, str]
    error_details: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CapabilityRegistry:
    """능력 레지스트리 - 동적 능력 등록 및 관리"""
    
    def __init__(self):
        self._capabilities: Dict[str, AgentCapability] = {}
        self._capability_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_capability(self, capability: AgentCapability, config: Dict[str, Any] = None):
        """능력 등록"""
        capability_name = capability.get_capability_name()
        self._capabilities[capability_name] = capability
        self._capability_configs[capability_name] = config or {}
    
    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """능력 조회"""
        return self._capabilities.get(name)
    
    def list_capabilities(self) -> List[str]:
        """사용 가능한 능력 목록"""
        return list(self._capabilities.keys())
    
    def get_capabilities_by_format(self, format_type: str) -> List[AgentCapability]:
        """형식별 능력 조회"""
        matching_capabilities = []
        for capability in self._capabilities.values():
            if format_type in capability.get_supported_formats():
                matching_capabilities.append(capability)
        return matching_capabilities


class FrameworkAdapter(ABC):
    """프레임워크 어댑터 추상 클래스"""
    
    @abstractmethod
    def get_framework_name(self) -> str:
        """프레임워크 이름 반환"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """프레임워크 초기화"""
        pass
    
    @abstractmethod
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """요청 처리"""
        pass
    
    @abstractmethod
    def get_framework_info(self) -> Dict[str, str]:
        """프레임워크 정보 반환"""
        pass


class ModularAgent(ABC):
    """완전 모듈화된 에이전트 기본 클래스"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.capability_registry = CapabilityRegistry()
        self.framework_adapter: Optional[FrameworkAdapter] = None
        
        # 기존 인프라 통합
        self.isolation_manager = UniversalIsolationManager()
        self.session_manager = UniversalSessionManager()
        self.communication_isolator = UniversalCommunicationIsolator()
        self.config_manager = ConfigManager()
        
        
        # 로깅 시스템
        self.logger = HybridLogger(config.agent_id)
        self.logging_manager = LoggingManager(self.logger)
        
        # 성능 추적
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
            "framework_specific_metrics": {}
        }
        
        # 초기화
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """에이전트 초기화"""
        # 프레임워크 어댑터 로드
        self.framework_adapter = await self._load_framework_adapter()
        
        # 기본 능력 등록
        await self._register_default_capabilities()
        
        # 사용자 정의 능력 로드
        await self._load_custom_capabilities()
        
        self.logger.info(f"{self.config.agent_id} 초기화 완료 - 프레임워크: {self.config.framework.value}")
    
    async def _load_framework_adapter(self) -> FrameworkAdapter:
        """프레임워크별 어댑터 동적 로드"""
        framework_name = self.config.framework.value
        
        try:
            # 동적 import로 프레임워크별 어댑터 로드
            adapter_module = importlib.import_module(f"adapters.{framework_name}_adapter")
            adapter_class = getattr(adapter_module, f"{framework_name.title()}Adapter")
            
            adapter = adapter_class()
            await adapter.initialize(self.config.custom_config)
            
            return adapter
            
        except ImportError as e:
            self.logger.error(f"프레임워크 어댑터 로드 실패: {framework_name} - {e}")
            # 폴백 어댑터 사용
            return await self._create_fallback_adapter()
    
    async def _create_fallback_adapter(self) -> FrameworkAdapter:
        """폴백 어댑터 생성"""
        class FallbackAdapter(FrameworkAdapter):
            def get_framework_name(self) -> str:
                return "fallback"
            
            async def initialize(self, config: Dict[str, Any]) -> bool:
                return True
            
            async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
                return ProcessingResponse(
                    request_id=request.request_id,
                    processed_content=request.content,
                    confidence_score=0.5,
                    quality_metrics={"fallback": 1.0},
                    processing_time=0.1,
                    framework_info={"name": "fallback", "version": "1.0"}
                )
            
            def get_framework_info(self) -> Dict[str, str]:
                return {"name": "fallback", "version": "1.0", "status": "active"}
        
        return FallbackAdapter()
    
    @abstractmethod
    async def _register_default_capabilities(self):
        """기본 능력 등록 (각 에이전트별로 구현)"""
        pass
    
    async def _load_custom_capabilities(self):
        """사용자 정의 능력 로드"""
        custom_capabilities_path = Path("capabilities") / f"{self.config.agent_id}_capabilities.json"
        
        if custom_capabilities_path.exists():
            try:
                with open(custom_capabilities_path, 'r', encoding='utf-8') as f:
                    capabilities_config = json.load(f)
                
                for cap_config in capabilities_config.get('capabilities', []):
                    await self._load_capability_from_config(cap_config)
                    
            except Exception as e:
                self.logger.warning(f"사용자 정의 능력 로드 실패: {e}")
    
    async def _load_capability_from_config(self, cap_config: Dict[str, Any]):
        """설정에서 능력 로드"""
        try:
            module_path = cap_config['module']
            class_name = cap_config['class']
            
            module = importlib.import_module(module_path)
            capability_class = getattr(module, class_name)
            
            capability = capability_class(**cap_config.get('init_args', {}))
            self.capability_registry.register_capability(capability, cap_config.get('config', {}))
            
            self.logger.info(f"사용자 정의 능력 로드됨: {capability.get_capability_name()}")
            
        except Exception as e:
            self.logger.error(f"능력 로드 실패: {cap_config} - {e}")
    
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """메인 처리 메서드"""
        start_time = time.time()
        
        try:
            # 요청 검증
            if not await self._validate_request(request):
                return self._create_error_response(request, "요청 검증 실패", start_time)
            
            # 언어 감지 및 설정
            await self._setup_language_processing(request)
            
            # 프레임워크별 처리
            response = await self.framework_adapter.process_request(request)
            
            # 후처리
            enhanced_response = await self._post_process_response(response, request)
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(True, time.time() - start_time)
            
            # 로깅
            await self._log_processing_success(request, enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            self._update_performance_metrics(False, time.time() - start_time)
            self.logger.error(f"처리 실패: {e}")
            return self._create_error_response(request, str(e), start_time)
    
    async def _validate_request(self, request: ProcessingRequest) -> bool:
        """요청 유효성 검사"""
        # 기본 필드 검증
        if not request.request_id or not request.content:
            return False
        
        # 격리 시스템을 통한 오염 검사
        if self.isolation_manager.is_contaminated(str(request.content), f"{self.config.agent_id}_input"):
            return False
        
        # 언어별 특수 검증 (LLM 기반으로 언어 중립적)
        if not await self._validate_content_quality(request):
            return False
        
        return True
    
    async def _validate_content_quality(self, request: ProcessingRequest) -> bool:
        """콘텐츠 품질 검증 (LLM 기반 언어 중립적)"""
        # 실제 구현에서는 LLM을 사용하여 언어에 관계없이 품질 검증
        return True
    
    async def _setup_language_processing(self, request: ProcessingRequest):
        """언어 처리 설정 (LLM 기반 자동 감지)"""
        if request.target_language == "auto":
            # LLM 기반 언어 감지
            detected_language = await self._detect_language_with_llm(request.content)
            request.target_language = detected_language
    
    async def _detect_language_with_llm(self, content: Any) -> str:
        """LLM 기반 언어 감지"""
        # 실제 구현에서는 LLM API 호출하여 언어 감지
        # 하드코딩 없이 LLM의 언어 이해 능력 활용
        return "ko"  # 임시값
    
    async def _post_process_response(self, response: ProcessingResponse, request: ProcessingRequest) -> ProcessingResponse:
        """응답 후처리"""
        # 품질 검증
        quality_check = await self._verify_response_quality(response, request)
        
        # 메타데이터 추가
        response.metadata.update({
            "agent_id": self.config.agent_id,
            "framework": self.config.framework.value,
            "quality_verified": quality_check,
            "language_processed": request.target_language
        })
        
        # 프레임워크 정보 추가
        response.framework_info = self.framework_adapter.get_framework_info()
        
        return response
    
    async def _verify_response_quality(self, response: ProcessingResponse, request: ProcessingRequest) -> bool:
        """응답 품질 검증"""
        # 최소 품질 요구사항 확인
        for metric, threshold in request.quality_requirements.items():
            if response.quality_metrics.get(metric, 0.0) < threshold:
                return False
        
        # 격리 시스템을 통한 출력 검증
        if self.isolation_manager.is_contaminated(str(response.processed_content), f"{self.config.agent_id}_output"):
            return False
        
        return True
    
    def _create_error_response(self, request: ProcessingRequest, error_msg: str, start_time: float) -> ProcessingResponse:
        """에러 응답 생성"""
        return ProcessingResponse(
            request_id=request.request_id,
            processed_content=None,
            confidence_score=0.0,
            quality_metrics={"error": 1.0},
            processing_time=time.time() - start_time,
            framework_info=self.framework_adapter.get_framework_info() if self.framework_adapter else {},
            error_details={"message": error_msg, "agent_id": self.config.agent_id},
            metadata={"error": True}
        )
    
    def _update_performance_metrics(self, success: bool, processing_time: float):
        """성능 메트릭 업데이트"""
        self.performance_metrics["total_requests"] += 1
        
        if success:
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["failed_requests"] += 1
        
        # 평균 처리 시간 업데이트
        total = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["avg_processing_time"]
        self.performance_metrics["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    async def _log_processing_success(self, request: ProcessingRequest, response: ProcessingResponse):
        """처리 성공 로깅"""
        await self.logging_manager.log_agent_response(
            agent_name=self.config.agent_id,
            agent_role=f"{self.config.framework.value} 기반 {self.__class__.__name__}",
            task_description=f"요청 처리: {request.content_type}",
            response_data={
                "confidence_score": response.confidence_score,
                "quality_metrics": response.quality_metrics,
                "processing_time": response.processing_time,
                "framework_info": response.framework_info
            },
            metadata={
                "request_id": request.request_id,
                "target_language": request.target_language,
                "agent_framework": self.config.framework.value
            }
        )
    
    async def communicate_with_agent(self, target_agent_id: str, data: Any, transfer_type: str = "result") -> Dict[str, Any]:
        """다른 에이전트와 통신"""
        request = DataTransferRequest(
            source_agent=self.config.agent_id,
            target_agent=target_agent_id,
            data=data,
            transfer_type=transfer_type,
            session_id=self.session_manager.create_session(),
            timestamp=time.time()
        )
        
        return self.communication_isolator.transfer_data(request)
    
    def add_capability(self, capability: AgentCapability, config: Dict[str, Any] = None):
        """런타임 능력 추가"""
        self.capability_registry.register_capability(capability, config)
        self.logger.info(f"새 능력 추가됨: {capability.get_capability_name()}")
    
    def get_capabilities(self) -> List[str]:
        """보유 능력 목록 반환"""
        return self.capability_registry.list_capabilities()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        success_rate = (
            self.performance_metrics["successful_requests"] / 
            max(self.performance_metrics["total_requests"], 1)
        )
        
        return {
            "agent_id": self.config.agent_id,
            "framework": self.config.framework.value,
            "performance": self.performance_metrics.copy(),
            "success_rate": success_rate,
            "capabilities_count": len(self.capability_registry.list_capabilities()),
            "framework_info": self.framework_adapter.get_framework_info() if self.framework_adapter else {}
        }


class AgentFactory:
    """에이전트 팩토리 - 설정 기반 에이전트 생성"""
    
    @staticmethod
    def create_agent(agent_type: str, config: AgentConfig) -> ModularAgent:
        """에이전트 타입에 따른 인스턴스 생성"""
        agent_classes = {
            "orchestrator": "OrchestratorAgent",
            "retriever": "RetrieverAgent", 
            "synthesizer": "SynthesizerAgent",
            "evaluator": "EvaluatorAgent"
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"지원하지 않는 에이전트 타입: {agent_type}")
        
        # 동적 import를 통한 에이전트 클래스 로드
        try:
            module = importlib.import_module(f"agents.{agent_type}_agent")
            agent_class = getattr(module, agent_classes[agent_type])
            return agent_class(config)
            
        except ImportError:
            raise ImportError(f"에이전트 모듈을 찾을 수 없음: agents.{agent_type}_agent")
    
    @staticmethod
    def create_from_config_file(config_path: str) -> List[ModularAgent]:
        """설정 파일에서 여러 에이전트 생성"""
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        agents = []
        for agent_config_data in configs.get('agents', []):
            config = AgentConfig(**agent_config_data)
            agent = AgentFactory.create_agent(
                agent_config_data['agent_type'], 
                config
            )
            agents.append(agent)
        
        return agents


class AgentOrchestrator:
    """에이전트 오케스트레이터 - 전체 시스템 관리"""
    
    def __init__(self):
        self.agents: Dict[str, ModularAgent] = {}
        self.agent_dependencies: Dict[str, List[str]] = {}
        self.communication_routes: Dict[str, Dict[str, str]] = {}
        
        # 로깅
        self.logger = HybridLogger("AgentOrchestrator")
    
    def register_agent(self, agent: ModularAgent, dependencies: List[str] = None):
        """에이전트 등록"""
        agent_id = agent.config.agent_id
        self.agents[agent_id] = agent
        self.agent_dependencies[agent_id] = dependencies or []
        
        self.logger.info(f"에이전트 등록됨: {agent_id} (프레임워크: {agent.config.framework.value})")
    
    def setup_communication_route(self, source_agent: str, target_agent: str, route_type: str):
        """통신 경로 설정"""
        if source_agent not in self.communication_routes:
            self.communication_routes[source_agent] = {}
        
        self.communication_routes[source_agent][target_agent] = route_type
    
    async def execute_workflow(self, initial_request: ProcessingRequest, workflow_config: Dict[str, Any]) -> Dict[str, ProcessingResponse]:
        """워크플로우 실행"""
        execution_plan = self._create_execution_plan(workflow_config)
        results = {}
        
        for stage in execution_plan:
            stage_results = await self._execute_stage(stage, initial_request, results)
            results.update(stage_results)
        
        return results
    
    def _create_execution_plan(self, workflow_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """실행 계획 생성"""
        # 워크플로우 설정에 따른 실행 순서 결정
        return workflow_config.get('execution_stages', [])
    
    async def _execute_stage(self, stage: Dict[str, Any], request: ProcessingRequest, previous_results: Dict[str, Any]) -> Dict[str, ProcessingResponse]:
        """단계 실행"""
        stage_results = {}
        
        # 병렬 실행 지원
        if stage.get('parallel', False):
            tasks = []
            for agent_id in stage['agents']:
                if agent_id in self.agents:
                    task = self._execute_agent_task(agent_id, request, previous_results)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            for agent_id, result in zip(stage['agents'], results):
                stage_results[agent_id] = result
        else:
            # 순차 실행
            for agent_id in stage['agents']:
                if agent_id in self.agents:
                    result = await self._execute_agent_task(agent_id, request, previous_results)
                    stage_results[agent_id] = result
        
        return stage_results
    
    async def _execute_agent_task(self, agent_id: str, request: ProcessingRequest, context: Dict[str, Any]) -> ProcessingResponse:
        """개별 에이전트 작업 실행"""
        agent = self.agents[agent_id]
        
        # 컨텍스트 기반 요청 수정
        modified_request = self._modify_request_with_context(request, context)
        
        # 에이전트 실행
        return await agent.process(modified_request)
    
    def _modify_request_with_context(self, original_request: ProcessingRequest, context: Dict[str, Any]) -> ProcessingRequest:
        """컨텍스트 기반 요청 수정"""
        # 이전 결과를 현재 요청에 통합
        modified_request = ProcessingRequest(
            request_id=original_request.request_id,
            content=original_request.content,
            content_type=original_request.content_type,
            target_language=original_request.target_language,
            processing_options=original_request.processing_options.copy(),
            quality_requirements=original_request.quality_requirements.copy(),
            metadata={**original_request.metadata, "context": context}
        )
        
        return modified_request
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        agent_statuses = {}
        
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_performance_summary()
        
        return {
            "total_agents": len(self.agents),
            "agent_statuses": agent_statuses,
            "communication_routes": self.communication_routes,
            "dependencies": self.agent_dependencies
        }


# 사용 예시 및 설정
def create_example_system():
    """예시 시스템 생성"""
    
    # 에이전트 설정들
    orchestrator_config = AgentConfig(
        agent_id="orchestrator_001",
        framework=AgentFramework.LANGGRAPH,
        capabilities=["workflow_management", "context_budgeting"],
        processing_mode=ProcessingMode.ASYNC,
        language_config={"default_language": "auto", "fallback_language": "en"}
    )
    
    retriever_config = AgentConfig(
        agent_id="retriever_001", 
        framework=AgentFramework.CREWAI,
        capabilities=["web_search", "api_integration", "multimodal_processing"],
        processing_mode=ProcessingMode.ASYNC,
        custom_config={"max_sources": 10, "search_depth": 3}
    )
    
    synthesizer_config = AgentConfig(
        agent_id="synthesizer_001",
        framework=AgentFramework.SEMANTIC_KERNEL,
        capabilities=["data_structuring", "format_conversion", "compression"],
        processing_mode=ProcessingMode.STREAMING,
        custom_config={"compression_ratio": 0.3, "output_formats": ["markdown", "json"]}
    )
    
    evaluator_config = AgentConfig(
        agent_id="evaluator_001",
        framework=AgentFramework.LANGCHAIN,
        capabilities=["hallucination_detection", "quality_assessment", "guardrails"],
        processing_mode=ProcessingMode.ASYNC,
        custom_config={"quality_thresholds": {"accuracy": 0.8, "relevance": 0.75}}
    )
    
    return {
        "configs": [orchestrator_config, retriever_config, synthesizer_config, evaluator_config],
        "workflow": {
            "execution_stages": [
                {"agents": ["orchestrator_001"], "parallel": False},
                {"agents": ["retriever_001"], "parallel": False},
                {"agents": ["synthesizer_001"], "parallel": False},
                {"agents": ["evaluator_001"], "parallel": False}
            ]
        }
    }


if __name__ == "__main__":
    # 시스템 초기화 예시
    system_config = create_example_system()
    
    # 오케스트레이터 생성
    orchestrator = AgentOrchestrator()