from typing import Dict, Any

# 기본 설정: LLM 모델, API 타임아웃, 캐시 등 에이전트의 핵심 동작을 제어합니다.
DEFAULT_CONFIG: Dict[str, Any] = {
    "llm_model": "gpt-4-turbo",  # 사용할 LLM 모델
    "temperature": 0.1,  # LLM의 생성 다양성 (낮을수록 일관된 답변)
    "request_timeout": 30,  # 외부 API 요청의 전체 타임아웃 (초)
    "llm_timeout": 60,  # LLM API 호출에 대한 타임아웃 (초)
    "cache_ttl": 3600,  # 캐시 유효 시간 (초, 1시간)
    "cache_size": 1000,  # 최대 캐시 항목 수
    "max_results": 5,  # 검색 시 가져올 최대 결과 수
    "verbose": False,  # 상세 로그 출력 여부
    "concurrent_limit": 5  # 동시 실행 가능한 작업 수
}

# 품질 임계값: 수집된 정보의 품질을 판단하는 기준값을 정의합니다.
QUALITY_THRESHOLDS: Dict[str, float] = {
    "relevance_threshold": 0.7,  # 관련성 점수가 이 값 이상이어야 유효한 정보로 간주
    "freshness_requirement": 0.8,  # 정보의 신선도 요구 수준 (0.0 ~ 1.0)
    "confidence_threshold": 0.6,  # 전체 신뢰도 점수의 최소 기준
    "quality_threshold": 0.5  # 전체 품질 점수의 최소 기준
}

# 검색 설정: 각 검색 작업의 타임아웃, 재시도 횟수 등 세부 동작을 제어합니다.
SEARCH_CONFIG: Dict[str, Any] = {
    "web_search_timeout": 15,  # 웹 검색 타임아웃 (초)
    "api_search_timeout": 10,  # API 검색 타임아웃 (초)
    "multimedia_timeout": 20,  # 멀티미디어 처리 타임아웃 (초)
    "max_retries": 3,  # 실패 시 최대 재시도 횟수
    "retry_delay": 1.0  # 재시도 사이의 대기 시간 (초)
}

# 로깅 설정: 애플리케이션의 로그 레벨, 형식, 출력 위치를 정의합니다.
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",  # 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"]  # 로그 출력 핸들러 (콘솔, 파일 등)
}

FEATURE_FLAGS = {
    "enable_multimedia": False,
    "enable_image_processing": False,
    "enable_video_processing": False,
    "enable_audio_processing": False
}