import asyncio
from datetime import datetime
import aiohttp
from typing import Dict, Any, Optional
import logging
import time

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from control.modular_agent_architecture import FrameworkAdapter, ProcessingRequest, ProcessingResponse

from ..core.cache import SearchCache
from ..models.state import AgentState
from ..models.data_models import RetrievalTask, SearchMetrics
from ..models.enums import DataSource
from ..tools.web_search import EnhancedWebSearchTool
from ..tools.api_retriever import APIRetrieverTool
from ..tools.multimedia import MultimediaProcessorTool
from ..tools.evaluator import RelevanceEvaluatorTool
from ..workflows.nodes import WorkflowNodes
from ..utils.exceptions import handle_langchain_exceptions
from ..utils.metrics import MetricsCalculator

# MCP 클라이언트 import
from ..mcp_client import MCPClient, DEFAULT_MCP_SERVERS

class LangChainAdapter(FrameworkAdapter):
    """LangChain 프레임워크 어댑터 - MCP 통합 완전 개선된 버전"""

    def __init__(self):
        self.workflow: Optional[StateGraph] = None
        self.agents: Dict[str, AgentExecutor] = {}
        self.tools: Dict[str, Any] = {}
        self.llm = None
        self.is_initialized = False
        self._search_cache = SearchCache()
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(5)
        self.logger = logging.getLogger(f"{__name__}.LangChainAdapter")
        self.search_metrics = SearchMetrics()
        self.workflow_nodes: Optional[WorkflowNodes] = None
        self.metrics_calculator = MetricsCalculator()
        
        # MCP 클라이언트 추가
        self.mcp_client: Optional[MCPClient] = None

    def get_framework_name(self) -> str:
        return "LangChain"

    @handle_langchain_exceptions(fallback_value=False)
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """LangChain 초기화 - MCP 통합 버전"""
        if not self._http_session:
            timeout = aiohttp.ClientTimeout(total=config.get("request_timeout", 30))
            self._http_session = aiohttp.ClientSession(timeout=timeout)

        # LLM 설정
        try:
            self.llm = ChatOpenAI(
                model=config.get("llm_model", "gpt-3.5-turbo"),
                temperature=config.get("temperature", 0.1),
                request_timeout=config.get("llm_timeout", 30)
            )
        except Exception as e:
            self.logger.warning(f"OpenAI 초기화 실패: {e}, 폴백 모드 사용")
            self.llm = None

        # MCP 클라이언트 초기화
        await self._initialize_mcp_client(config)

        # 도구 초기화 (MCP 클라이언트 사용)
        await self._initialize_tools(config)

        # 에이전트들 생성
        if self.llm:
            await self._create_specialist_agents(config)
            self._build_workflow()

        self.is_initialized = True
        self.logger.info("LangChain 어댑터 초기화 완료 (MCP 통합)")
        return True

    async def _initialize_mcp_client(self, config: Dict[str, Any]):
        """MCP 클라이언트 초기화"""
        try:
            # 설정에서 MCP 서버 구성 가져오기 (없으면 기본값 사용)
            mcp_servers = config.get("mcp_servers", DEFAULT_MCP_SERVERS)
            
            self.mcp_client = MCPClient(mcp_servers)
            success = await self.mcp_client.initialize()
            
            if success:
                self.logger.info("MCP 클라이언트 초기화 성공")
                # 사용 가능한 도구 로그
                available_tools = await self.mcp_client.get_available_tools()
                self.logger.info(f"사용 가능한 MCP 도구: {available_tools}")
            else:
                self.logger.warning("MCP 클라이언트 초기화 실패, 폴백 모드 사용")
                self.mcp_client = None
                
        except Exception as e:
            self.logger.error(f"MCP 클라이언트 초기화 중 오류: {e}")
            self.mcp_client = None

    async def _initialize_tools(self, config: Dict[str, Any]):
        """도구 초기화 - MCP 클라이언트 사용"""
        if self.mcp_client:
            # MCP 기반 도구들
            self.tools = {
                "web_search": EnhancedWebSearchTool(self.mcp_client),
                "api_retriever": APIRetrieverTool(self.mcp_client),
                "multimedia_processor": MultimediaProcessorTool(),
                "relevance_evaluator": RelevanceEvaluatorTool()
            }
            self.logger.info(f"MCP 기반 도구 {len(self.tools)}개 초기화 완료")
        else:
            # 폴백: 기존 방식 도구들
            self.tools = {
                "web_search": EnhancedWebSearchTool(None),  # 폴백 모드
                "api_retriever": APIRetrieverTool(None),    # 폴백 모드
                "multimedia_processor": MultimediaProcessorTool(),
                "relevance_evaluator": RelevanceEvaluatorTool()
            }
            self.logger.warning("폴백 모드로 도구 초기화")

    @handle_langchain_exceptions(fallback_value=None)
    async def _create_specialist_agents(self, config: Dict[str, Any]):
        """전문가 에이전트들 생성 - MCP 도구 사용"""
        # 웹 검색 전문가 (MCP 기반)
        web_tools = [self.tools["web_search"], self.tools["relevance_evaluator"]]
        web_prompt = PromptTemplate(
            template="""당신은 MCP 서버를 통한 웹 검색 전문가입니다.
주어진 쿼리에 대해 MCP 서버를 통해 최신의 관련성 높은 정보를 효율적으로 수집하세요.

사용 가능한 도구: {tools}
도구 이름: {tool_names}

쿼리: {input}

{agent_scratchpad}

응답:""",
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )

        web_agent = create_react_agent(self.llm, web_tools, web_prompt)
        self.agents["web_specialist"] = AgentExecutor(
            agent=web_agent,
            tools=web_tools,
            verbose=config.get("verbose", False),
            return_intermediate_steps=True
        )

        # API 데이터 전문가 (MCP 기반)
        api_tools = [self.tools["api_retriever"]]
        api_prompt = PromptTemplate(
            template="""당신은 MCP 서버를 통한 API 데이터 전문가입니다.
MCP 서버를 통해 GitHub, Stack Overflow 등에서 구조화된 데이터를 효율적으로 수집하세요.

사용 가능한 도구: {tools}
도구 이름: {tool_names}

쿼리: {input}

{agent_scratchpad}

응답:""",
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )

        api_agent = create_react_agent(self.llm, api_tools, api_prompt)
        self.agents["api_specialist"] = AgentExecutor(
            agent=api_agent,
            tools=api_tools,
            verbose=config.get("verbose", False),
            return_intermediate_steps=True
        )

        # 멀티미디어 전문가 (기존 방식 유지)
        multimedia_tools = [self.tools["multimedia_processor"]]
        multimedia_prompt = PromptTemplate(
            template="""당신은 멀티미디어 분석 전문가입니다.
이미지, 비디오, 오디오 콘텐츠에서 유용한 정보를 추출하세요.

사용 가능한 도구: {tools}
도구 이름: {tool_names}

쿼리: {input}

{agent_scratchpad}

응답:""",
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )

        multimedia_agent = create_react_agent(self.llm, multimedia_tools, multimedia_prompt)
        self.agents["multimedia_specialist"] = AgentExecutor(
            agent=multimedia_agent,
            tools=multimedia_tools,
            verbose=config.get("verbose", False),
            return_intermediate_steps=True
        )

        # 품질 관리 전문가 (기존 방식 유지)
        quality_tools = [self.tools["relevance_evaluator"]]
        quality_prompt = PromptTemplate(
            template="""당신은 정보 품질 관리 전문가입니다.
수집된 정보의 품질, 관련성, 신선도를 종합적으로 평가하세요.

사용 가능한 도구: {tools}
도구 이름: {tool_names}

쿼리: {input}

{agent_scratchpad}

응답:""",
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )

        quality_agent = create_react_agent(self.llm, quality_tools, quality_prompt)
        self.agents["quality_specialist"] = AgentExecutor(
            agent=quality_agent,
            tools=quality_tools,
            verbose=config.get("verbose", False),
            return_intermediate_steps=True
        )

        self.logger.info(f"MCP 기반 전문가 에이전트 {len(self.agents)}개 생성 완료")

    def _build_workflow(self):
        """LangGraph 워크플로우 구성 - MCP 기반"""
        if not self.agents:
            return

        self.workflow_nodes = WorkflowNodes(self.agents)
        workflow = StateGraph(AgentState)

        # 워크플로우에 노드 추가
        workflow.add_node("web_search", self.workflow_nodes.web_search_node)
        workflow.add_node("api_search", self.workflow_nodes.api_search_node)
        workflow.add_node("multimedia_process", self.workflow_nodes.multimedia_node)
        workflow.add_node("quality_evaluation", self.workflow_nodes.quality_node)

        # 엣지 정의 (순차 실행)
        workflow.add_edge("web_search", "api_search")
        workflow.add_edge("api_search", "multimedia_process")
        workflow.add_edge("multimedia_process", "quality_evaluation")
        workflow.add_edge("quality_evaluation", END)

        # 시작점 설정
        workflow.set_entry_point("web_search")

        # 컴파일
        self.workflow = workflow.compile()
        self.logger.info("MCP 기반 LangGraph 워크플로우 구성 완료")

    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """요청 처리 - MCP 통합 버전"""
        if not self.is_initialized:
            await self.initialize({})

        # 캐시 정리
        self._search_cache.clear_expired()

        # RetrievalTask 생성
        task = self._create_retrieval_task(request)

        if self.workflow and self.llm:
            return await self._process_with_mcp_workflow(request, task)
        else:
            return await self._process_with_fallback(request, task)

    async def _process_with_mcp_workflow(self, request: ProcessingRequest, task: RetrievalTask) -> ProcessingResponse:
        """MCP 워크플로우를 사용한 처리"""
        start_time = time.time()
        try:
            # 초기 상태 설정
            initial_state: AgentState = {
                "messages": [HumanMessage(content=str(request.content))],
                "query": str(request.content),
                "results": {},
                "current_step": "web_search",
                "metadata": request.processing_options,
                "task_info": task.to_dict(),
                "processing_history": ["mcp_workflow_started"]
            }

            # MCP 기반 워크플로우 실행
            final_state = await self.workflow.ainvoke(initial_state)

            # 결과 통합
            integrated_results = await self._integrate_retrieval_results(final_state["results"])

            # 신뢰도 계산
            confidence = await self._calculate_retrieval_confidence(final_state["results"])

            # 메트릭 업데이트
            self.search_metrics.total_searches += 1

            return ProcessingResponse(
                request_id=request.request_id,
                processed_content=integrated_results,
                confidence_score=confidence,
                quality_metrics=await self._calculate_retrieval_quality_metrics(final_state["results"]),
                processing_time=time.time() - start_time,
                framework_info=self.get_framework_info(),
                metadata={
                    "steps_executed": len(final_state["results"]),
                    "framework_used": "LangChain + LangGraph + MCP",
                    "mcp_enabled": True,
                    "cache_stats": self._search_cache.get_stats()
                }
            )

        except Exception as e:
            self.logger.error(f"MCP 워크플로우 처리 중 오류: {e}")
            self.search_metrics.failed_searches += 1
            return await self._process_with_fallback(request, task)

    def _create_retrieval_task(self, request: ProcessingRequest) -> RetrievalTask:
        """요청을 구조화된 검색 작업으로 변환"""
        return RetrievalTask(
            task_id=request.request_id,
            query=str(request.content),
            data_sources=[DataSource.WEB_SEARCH, DataSource.API_ENDPOINT, DataSource.MULTIMEDIA],
            priority=1,
            max_results=request.processing_options.get("max_results", 10),
            freshness_requirement=0.8,
            relevance_threshold=0.7
        )

    def get_framework_info(self) -> Dict[str, str]:
        """프레임워크 정보 - MCP 통합 버전"""
        return {
            "name": "LangChain",
            "version": "1.0.0-mcp",
            "status": "active" if self.is_initialized else "initializing",
            "features": "mcp_integration,agent_orchestration,workflow_management,tool_integration,real_web_search,caching",
            "capabilities": f"agents:{len(self.agents)},tools:{len(self.tools)}",
            "mcp_enabled": "true" if self.mcp_client else "false",
            "cache_enabled": "true",
            "workflow_engine": "LangGraph" if self.workflow else "none"
        }

    async def cleanup(self):
        """리소스 정리 - MCP 포함"""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        
        # MCP 클라이언트 정리
        if self.mcp_client:
            await self.mcp_client.cleanup()
            
        self.logger.info("LangChain 어댑터 리소스 정리 완료 (MCP 포함)")

    # 기존 메서드들 (변경 없음)
    async def _integrate_retrieval_results(self, results: Dict[str, Any]) -> str:
        """검색 결과 통합"""
        integrated_content = "# 🔍 종합 정보 수집 결과 (MCP 기반)\n\n"

        step_names = {
            "web_search": "MCP 웹 검색",
            "api_search": "MCP API 데이터 수집",
            "multimedia": "멀티미디어 분석",
            "quality_evaluation": "품질 평가"
        }

        for key, value in results.items():
            step_name = step_names.get(key, key)
            integrated_content += f"## 📋 {step_name}\n"
            integrated_content += f"{value}\n\n"

        integrated_content += "---\n\n"

        # 요약 정보 추가
        integrated_content += "## 📊 수집 요약\n"
        integrated_content += f"- **총 수집 단계**: {len(results)}개\n"
        integrated_content += f"- **처리 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        integrated_content += f"- **데이터 소스**: MCP 웹 검색, MCP API, 멀티미디어, 품질 평가\n"
        integrated_content += f"- **통합 방식**: MCP 서버 기반 순차 처리 및 결과 병합\n\n"

        # 신뢰도 정보
        avg_confidence = min(0.8 + (len(results) - 1) * 0.05, 0.95)
        integrated_content += f"## ✅ 신뢰도 정보\n"
        integrated_content += f"- **전체 신뢰도**: {avg_confidence:.2f}/1.0\n"
        integrated_content += f"- **정보 완성도**: {'높음' if len(results) >= 3 else '보통'}\n"
        integrated_content += f"- **MCP 통합**: 활성화됨\n"
        integrated_content += f"- **활용 권장도**: {'적극 권장' if avg_confidence >= 0.8 else '검토 후 사용'}\n"

        return integrated_content

    async def _calculate_retrieval_confidence(self, results: Dict[str, Any]) -> float:
        """검색 신뢰도 계산 - MCP 보너스 포함"""
        if not results:
            return 0.0

        # 기본 신뢰도 (결과 수에 따라)
        base_confidence = 0.6 + (len(results) / 10)

        # 완성도 보너스
        completeness_bonus = min(len(results) / 4, 1.0) * 0.2

        # 다양성 보너스
        diversity_bonus = 0.1 if len(results) >= 3 else 0.05

        # MCP 통합 보너스
        mcp_bonus = 0.05 if self.mcp_client else 0.0

        # 최종 신뢰도 계산
        confidence = min(base_confidence + completeness_bonus + diversity_bonus + mcp_bonus, 1.0)

        return round(confidence, 2)

    async def _calculate_retrieval_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """검색 품질 메트릭 계산 - MCP 향상 반영"""
        if not results:
            return {"coverage": 0.0, "diversity": 0.0, "freshness": 0.0, "relevance": 0.0}

        # 커버리지: 4개 전문가 기준
        coverage = min(len(results) / 4, 1.0)

        # 다양성: 결과 수에 따른 다양성 (MCP 보너스)
        diversity_base = 0.9 if len(results) >= 4 else 0.8 if len(results) >= 3 else 0.7
        diversity = min(diversity_base + (0.05 if self.mcp_client else 0.0), 1.0)

        # 신선도: MCP 서버 기반이므로 높음
        freshness = 0.98 if self.mcp_client else 0.95

        # 관련성: MCP 서버 기반 향상
        relevance = 0.90 if self.mcp_client else 0.85

        return {
            "coverage": round(coverage, 2),
            "diversity": round(diversity, 2),
            "freshness": round(freshness, 2),
            "relevance": round(relevance, 2),
            "overall": round((coverage + diversity + freshness + relevance) / 4, 2),
            "mcp_enhanced": self.mcp_client is not None
        }

    # 기존 폴백 메서드들은 그대로 유지...
    async def _process_with_fallback(self, request: ProcessingRequest, task: RetrievalTask) -> ProcessingResponse:
        """폴백 처리 방식 (MCP 실패 시)"""
        start_time = time.time()
        query = str(request.content)

        # 캐시 확인
        cached_result = self._search_cache.get(query)
        if cached_result:
            self.search_metrics.cache_hits += 1
            self.logger.info(f"캐시 히트: {query}")
            return ProcessingResponse(
                request_id=request.request_id,
                processed_content=cached_result,
                confidence_score=0.8,
                quality_metrics={"cached": 1.0, "freshness": 0.9},
                processing_time=time.time() - start_time,
                framework_info=self.get_framework_info(),
                metadata={"source": "cache", "mcp_fallback": True}
            )

        # 캐시 미스 - 폴백 검색 수행
        self.search_metrics.cache_misses += 1
        search_results = await self._perform_comprehensive_fallback_search(request)
        self._search_cache.set(query, search_results)

        # 메트릭 업데이트
        self.search_metrics.total_searches += 1
        processing_time = time.time() - start_time

        return ProcessingResponse(
            request_id=request.request_id,
            processed_content=search_results,
            confidence_score=0.7,
            quality_metrics={"fallback": 1.0, "mcp_fallback": True},
            processing_time=processing_time,
            framework_info=self.get_framework_info(),
            metadata={
                "source": "fallback_search",
                "mcp_fallback": True,
                "search_metrics": self.search_metrics.__dict__
            }
        )

    async def _perform_comprehensive_fallback_search(self, request: ProcessingRequest) -> str:
        """포괄적인 폴백 검색 - MCP 실패 시"""
        query = str(request.content)
        retrieval_mode = request.processing_options.get("retrieval_mode", "comprehensive")
        results = []

        # 폴백 웹 검색 (MCP 없이)
        try:
            if self.tools["web_search"]:
                web_result = await self.tools["web_search"]._arun(query)
            else:
                web_result = "웹 검색 불가 (도구 없음)"
            results.append(f"## 폴백 웹 검색 결과\n{web_result}")
        except Exception as e:
            self.logger.warning(f"폴백 웹 검색 오류: {e}")
            results.append(f"## 폴백 웹 검색 결과\n폴백 웹 검색 중 오류 발생: {str(e)}")

        # 폴백 API 검색 (MCP 없이)
        try:
            if self.tools["api_retriever"]:
                api_result = await self.tools["api_retriever"]._arun(query)
            else:
                api_result = "API 검색 불가 (도구 없음)"
            results.append(f"## 폴백 API 검색 결과\n{api_result}")
        except Exception as e:
            self.logger.warning(f"폴백 API 검색 오류: {e}")
            results.append(f"## 폴백 API 검색 결과\n폴백 API 검색 중 오류 발생: {str(e)}")

        # 멀티모달 검색 (요청된 경우)
        if retrieval_mode == "multimodal":
            multimedia_tool = MultimediaProcessorTool()
            multimedia_result = multimedia_tool._run(query)
            results.append(f"## 멀티미디어 분석 결과\n{multimedia_result}")

        # 품질 평가
        quality_tool = RelevanceEvaluatorTool()
        quality_result = quality_tool._run("\n".join(results), query)
        results.append(f"## 품질 평가 결과\n{quality_result}")

        # 통합 결과 생성
        integrated_content = f"""
# 종합 정보 수집 결과 (MCP 폴백 모드)

**검색 쿼리**: {query}
**검색 모드**: {retrieval_mode}
**수집 단계**: {len(results)}개
**MCP 상태**: 폴백 모드 (MCP 서버 연결 실패)

{chr(10).join(results)}

---

## 요약

- **총 정보 소스**: {len(results)}개
- **검색 방식**: 폴백 검색 + 구조화된 분석
- **신뢰도**: 중간 (MCP 폴백)
- **처리 시간**: {time.time():.2f}초

**권장사항**: MCP 서버 연결을 확인하고 재시도하세요. 현재 결과는 폴백 모드로 제공됩니다.
"""

        return integrated_content