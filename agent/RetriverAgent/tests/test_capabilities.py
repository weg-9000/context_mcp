import pytest
from unittest.mock import patch, AsyncMock

from ..capabilities import (
    WebSearchCapability,
    MultimodalProcessingCapability,
    RelevanceEvaluationCapability
)

@pytest.mark.asyncio
async def test_web_search_capability():
    """WebSearchCapability 실행 테스트"""
    capability = WebSearchCapability()
    result = await capability.execute("pytest 사용법", {"max_results": 3})

    assert result["query"] == "pytest 사용법"
    assert len(result["results"]) == 3
    assert "error" not in result

@pytest.mark.asyncio
async def test_multimodal_processing_capability_image():
    """MultimodalProcessingCapability 이미지 처리 테스트"""
    capability = MultimodalProcessingCapability()
    result = await capability.execute("image_data", {"content_type": "image"})

    assert result["content_type"] == "image"
    assert "extracted_text" in result
    assert "objects_detected" in result

@pytest.mark.asyncio
async def test_multimodal_processing_capability_unsupported():
    """MultimodalProcessingCapability 지원하지 않는 타입 처리 테스트"""
    capability = MultimodalProcessingCapability()
    # 'document'는 현재 시뮬레이션에서 'else' 분기에 해당
    result = await capability.execute("document_data", {"content_type": "document"})

    assert result["content_type"] == "document"
    assert "extracted_text" in result

@pytest.mark.asyncio
async def test_relevance_evaluation_capability():
    """RelevanceEvaluationCapability 실행 테스트"""
    capability = RelevanceEvaluationCapability()
    input_data = {
        "content": "RetrieverAgent는 LangChain을 사용합니다.",
        "query": "LangChain 기반 에이전트"
    }
    result = await capability.execute(input_data, {})

    assert "relevance_score" in result
    assert result["relevance_score"] > 0.5
    assert result["recommendation"] == "include"
    assert "detailed_analysis" in result