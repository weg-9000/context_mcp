import pytest
from unittest.mock import patch, AsyncMock

from modular_agent_architecture import AgentConfig, ProcessingRequest, AgentFramework, ProcessingResponse
from ..core.agent import RetrieverAgent
from ..models.enums import RetrievalMode

@pytest.fixture
def agent_config():
    """테스트용 에이전트 설정 Fixture"""
    return AgentConfig(
        agent_id="test_retriever_agent",
        framework=AgentFramework.LANGCHAIN,
        processing_options={
            "llm_model": "gpt-3.5-turbo-mock",
            "temperature": 0.0,
            "cache_ttl": 3600,
            "verbose": False
        }
    )

@pytest.fixture
def test_request():
    """테스트용 처리 요청 Fixture"""
    return ProcessingRequest(
        request_id="test_req_123",
        content="LangChain의 ReAct 패턴이란 무엇인가?",
        processing_options={
            "retrieval_mode": RetrievalMode.COMPREHENSIVE.value,
            "max_results": 3
        }
    )

def test_agent_initialization(agent_config):
    """에이전트 초기화 및 능력 등록 테스트"""
    agent = RetrieverAgent(agent_config)
    assert agent.config.framework == AgentFramework.LANGCHAIN
    assert agent.framework_adapter is not None
    assert "web_search" in agent.capabilities
    assert "multimodal_processing" in agent.capabilities
    assert "relevance_evaluation" in agent.capabilities

def test_invalid_framework_raises_error():
    """지원하지 않는 프레임워크 설정 시 ValueError 발생 테스트"""
    with pytest.raises(ValueError, match="LangChain 프레임워크만 지원합니다"):
        invalid_config = AgentConfig(
            agent_id="invalid_agent",
            framework="UnsupportedFramework", # 잘못된 프레임워크
            processing_options={}
        )
        RetrieverAgent(invalid_config)

@pytest.mark.asyncio
async def test_agent_async_initialize(agent_config):
    """에이전트 비동기 초기화 성공 테스트"""
    agent = RetrieverAgent(agent_config)
    # framework_adapter의 initialize 메서드를 모킹하여 항상 True를 반환하도록 설정
    with patch.object(agent.framework_adapter, 'initialize', new_callable=AsyncMock) as mock_init:
        mock_init.return_value = True
        result = await agent.initialize()
        assert result is True
        mock_init.assert_called_once()

@pytest.mark.asyncio
async def test_process_request_flow(agent_config, test_request):
    """요청 처리 흐름 테스트"""
    agent = RetrieverAgent(agent_config)

    # Mock 응답 객체 생성
    mock_response = ProcessingResponse(
        request_id=test_request.request_id,
        processed_content="Mocked response content",
        confidence_score=0.9,
        quality_metrics={"coverage": 1.0},
        processing_time=0.5,
        metadata={}
    )

    # process_request 메서드를 모킹
    with patch.object(agent.framework_adapter, 'process_request', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = mock_response
        response = await agent.process_request(test_request)

        # 결과 검증
        mock_process.assert_called_once_with(test_request)
        assert response.request_id == test_request.request_id
        assert "agent_type" in response.metadata
        assert response.metadata["agent_type"] == "RetrieverAgent"
        assert agent.performance_metrics.total_searches == 1

def test_get_agent_info(agent_config):
    """에이전트 정보 조회 메서드 테스트"""
    agent = RetrieverAgent(agent_config)
    info = agent.get_agent_info()

    assert info["name"] == "RetrieverAgent"
    assert info["framework"] == "LangChain"
    assert info["architecture"] == "modular"
    assert "capabilities" in info
    assert len(info["capabilities"]) >= 3