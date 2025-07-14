import pytest
from unittest.mock import AsyncMock, MagicMock

from ..tools import (
    EnhancedWebSearchTool,
    APIRetrieverTool,
    MultimediaProcessorTool,
    RelevanceEvaluatorTool
)

@pytest.mark.asyncio
async def test_enhanced_web_search_tool_success():
    """EnhancedWebSearchTool 성공 케이스 테스트"""
    mock_session = AsyncMock()
    mock_response = mock_session.get.return_value.__aenter__.return_value
    mock_response.status = 200
    mock_response.json.return_value = {
        "AbstractText": "이것은 웹 검색 결과의 요약입니다.",
        "RelatedTopics": [{"Text": "관련 주제 1"}]
    }

    tool = EnhancedWebSearchTool(http_session=mock_session)
    result = await tool._arun("test query")

    assert "이것은 웹 검색 결과의 요약입니다." in result
    assert "관련 주제 1" in result
    mock_session.get.assert_called_once()

@pytest.mark.asyncio
async def test_enhanced_web_search_tool_api_failure():
    """EnhancedWebSearchTool API 실패 시 폴백 테스트"""
    mock_session = AsyncMock()
    mock_response = mock_session.get.return_value.__aenter__.return_value
    mock_response.status = 500  # 서버 오류 시뮬레이션

    tool = EnhancedWebSearchTool(http_session=mock_session)
    result = await tool._arun("test query")

    assert "시뮬레이션 검색 결과" in result

def test_api_retriever_tool():
    """APIRetrieverTool 실행 테스트"""
    tool = APIRetrieverTool()
    result = tool._run("python api")
    assert "GitHub API" in result
    assert "Stack Overflow API" in result

def test_multimedia_processor_tool():
    """MultimediaProcessorTool 실행 테스트"""
    tool = MultimediaProcessorTool()
    result = tool._run("이미지 분석")
    assert "멀티미디어 처리 결과" in result
    assert "이미지: 스크린샷 및 다이어그램 분석" in result

def test_relevance_evaluator_tool():
    """RelevanceEvaluatorTool 실행 테스트"""
    tool = RelevanceEvaluatorTool()
    result = tool._run(content="langchain is a framework", query="what is langchain")
    assert "평가 점수" in result
    assert "권장사항" in result
    assert "포함" in result