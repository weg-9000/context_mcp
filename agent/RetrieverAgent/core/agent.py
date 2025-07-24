import time
from typing import Dict, Any

from control.modular_agent_architecture import ModularAgent, AgentConfig, ProcessingRequest, ProcessingResponse, AgentFramework

from ..core.adapter import LangChainAdapter
from ..core.cache import SearchCache
from ..capabilities.web_search import WebSearchCapability
from ..capabilities.multimodal import MultimodalProcessingCapability
from ..capabilities.relevance import RelevanceEvaluationCapability
from ..models.enums import RetrievalMode
from ..models.data_models import SearchMetrics
from ..utils.exceptions import handle_langchain_exceptions

class RetrieverAgent(ModularAgent):
    """LangChain 기반 정보 수집 에이전트 - 완전 모듈화된 버전"""

    def __init__(self, config: AgentConfig):
        if config.framework != AgentFramework.LANGCHAIN:
            raise ValueError(f"RetrieverAgent는 LangChain 프레임워크만 지원합니다. 현재: {config.framework}")

        super().__init__(config)
        self.framework_adapter = LangChainAdapter()
        self._register_capabilities()
        self.search_cache = SearchCache(
            ttl=config.processing_options.get("cache_ttl", 3600),
            max_size=config.processing_options.get("cache_size", 1000)
        )
        self.performance_metrics = SearchMetrics()

    def _register_capabilities(self):
        """능력 등록"""
        self.register_capability("web_search", WebSearchCapability())
        self.register_capability("multimodal_processing", MultimodalProcessingCapability())
        self.register_capability("relevance_evaluation", RelevanceEvaluationCapability())

    @handle_langchain_exceptions(fallback_value=False)
    async def initialize(self) -> bool:
        """에이전트 초기화"""
        adapter_config = {
            "llm_model": self.config.processing_options.get("llm_model", "gpt-3.5-turbo"),
            "temperature": self.config.processing_options.get("temperature", 0.1),
            "request_timeout": self.config.processing_options.get("request_timeout", 30),
            "verbose": self.config.processing_options.get("verbose", False),
            "cache_ttl": self.config.processing_options.get("cache_ttl", 3600),
            "cache_size": self.config.processing_options.get("cache_size", 1000)
        }
        success = await self.framework_adapter.initialize(adapter_config)
        if success:
            self.logger.info("RetrieverAgent 초기화 완료")
            return True
        else:
            self.logger.warning("RetrieverAgent 폴백 모드로 초기화")
            return True

    @handle_langchain_exceptions(fallback_value=None)
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """요청 처리"""
        start_time = time.time()
        self.performance_metrics.total_searches += 1
        retrieval_mode = request.processing_options.get("retrieval_mode", RetrievalMode.COMPREHENSIVE.value)
        response = await self.framework_adapter.process_request(request)
        
        processing_time = time.time() - start_time
        self._update_metrics(processing_time, success=True)
        
        if response and hasattr(response, 'metadata'):
            response.metadata.update({
                "agent_type": "RetrieverAgent",
                "retrieval_mode": retrieval_mode,
                "performance_metrics": self.performance_metrics.__dict__
            })
        
        self.logger.info(f"요청 처리 완료: {request.request_id}, 시간: {processing_time:.2f}초")
        return response

    def _update_metrics(self, processing_time: float, success: bool = True):
        """메트릭 업데이트"""
        if success:
            total = self.performance_metrics.total_searches
            current_avg = self.performance_metrics.average_response_time
            if total == 1:
                self.performance_metrics.average_response_time = processing_time
            else:
                new_avg = ((current_avg * (total - 1)) + processing_time) / total
                self.performance_metrics.average_response_time = round(new_avg, 3)
        else:
            self.performance_metrics.failed_searches += 1
        self.performance_metrics.last_updated = time.time()

    def get_agent_info(self) -> Dict[str, Any]:
        """에이전트 정보"""
        return {
            "name": "RetrieverAgent",
            "version": "1.0.0",
            "framework": "LangChain",
            "description": "모듈화된 LangChain 기반 정보 수집 에이전트",
            "capabilities": [
                "web_search",
                "multimodal_processing", 
                "relevance_evaluation",
                "real_time_search",
                "caching",
                "quality_assessment"
            ],
            "supported_modes": [mode.value for mode in RetrievalMode],
            "performance_metrics": self.performance_metrics.__dict__,
            "architecture": "modular",
            "last_updated": time.time()
        }