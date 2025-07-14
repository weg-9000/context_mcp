import time
from typing import Dict, Any

from modular_agent_architecture import ProcessingResponse, ProcessingRequest

def create_error_response(request: ProcessingRequest, error: Exception, start_time: float) -> ProcessingResponse:
    """표준화된 오류 응답 생성"""
    return ProcessingResponse(
        request_id=request.request_id,
        processed_content=f"검색 중 오류가 발생했습니다: {str(error)}",
        confidence_score=0.0,
        quality_metrics={"error": 1.0},
        processing_time=time.time() - start_time,
        framework_info={"name": "LangChain", "status": "error"},
        error_details={
            "message": str(error),
            "type": type(error).__name__,
            "framework": "LangChain"
        },
        metadata={
            "agent_type": "RetrieverAgent",
            "error_occurred": True
        }
    )

def integrate_retrieval_results(results: Dict[str, Any]) -> str:
    """검색 결과를 통합하여 최종 보고서 형식으로 만듭니다."""
    integrated_content = "# 🔍 종합 정보 수집 결과\n\n"

    step_names = {
        "web_search": "웹 검색",
        "api_search": "API 데이터 수집",
        "multimedia": "멀티미디어 분석",
        "quality_evaluation": "품질 평가"
    }

    for key, value in results.items():
        step_name = step_names.get(key, key.replace("_", " ").title())
        integrated_content += f"## 📋 {step_name}\n{value}\n\n"

    integrated_content += "---\n\n## 📊 수집 요약\n"
    integrated_content += f"- **총 수집 단계**: {len(results)}개\n"
    integrated_content += f"- **데이터 소스**: {', '.join(step_names.get(k, k) for k in results.keys())}\n"

    return integrated_content