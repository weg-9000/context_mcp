"""환경 변수 기반 설정 관리"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# 프로젝트 루트에서 .env 파일 로드
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

# 환경 변수에서 설정 로드
def get_llm_config() -> Dict[str, Any]:
    """LLM 설정을 환경 변수에서 로드"""
    return {
        "model": os.getenv("LLM_MODEL"),
        "provider": os.getenv("LLM_PROVIDER"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "2000")),
        "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "30"))
    }

def get_search_config() -> Dict[str, Any]:
    """검색 설정을 환경 변수에서 로드"""
    return {
        "max_results": int(os.getenv("MAX_RESULTS", "5")),
        "web_search_timeout": int(os.getenv("WEB_SEARCH_TIMEOUT", "15")),
        "api_search_timeout": int(os.getenv("API_SEARCH_TIMEOUT", "10")),
        "max_retries": int(os.getenv("MAX_RETRIES", "3"))
    }

def get_cache_config() -> Dict[str, Any]:
    """캐시 설정을 환경 변수에서 로드"""
    return {
        "ttl": int(os.getenv("CACHE_TTL", "3600")),
        "size": int(os.getenv("CACHE_SIZE", "1000"))
    }

def get_logging_config() -> Dict[str, Any]:
    """로깅 설정을 환경 변수에서 로드"""
    return {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "verbose": os.getenv("VERBOSE", "false").lower() == "true"
    }

def get_api_tokens() -> Dict[str, str]:
    """API 토큰들을 환경 변수에서 로드"""
    return {
        "github_token": os.getenv("GITHUB_TOKEN", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "google_api_key": os.getenv("GOOGLE_API_KEY", "")
    }

def validate_environment() -> bool:
    """필수 환경 변수 검증"""
    api_tokens = get_api_tokens()
    
    
    if not api_tokens["openai_api_key"]:
        print(" LLM API 키가 필요합니다. OPENAI_API_KEY를 설정하세요.")
        return False
    
    return True

def get_all_config() -> Dict[str, Any]:
    """모든 설정을 통합하여 반환"""
    return {
        "llm": get_llm_config(),
        "search": get_search_config(),
        "cache": get_cache_config(),
        "logging": get_logging_config(),
        "api_tokens": get_api_tokens()
    }

SEARCH_CONFIG = get_search_config()
LOGGING_CONFIG = get_logging_config()
DEFAULT_CONFIG = get_all_config()
QUALITY_THRESHOLDS = {
    "relevance": 0.7,
    "freshness": 0.8,
    "completeness": 0.75
}
