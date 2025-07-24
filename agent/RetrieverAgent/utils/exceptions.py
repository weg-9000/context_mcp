import logging
from functools import wraps
import asyncio
from typing import Any

from langchain.schema import LangChainException
from langchain.tools.base import ToolException

class RetrieverAgentException(Exception):
    """RetrieverAgent 전용 예외"""
    pass

class CacheException(RetrieverAgentException):
    """캐시 관련 예외"""
    pass

class WorkflowException(RetrieverAgentException):
    """워크플로우 관련 예외"""
    pass

def handle_langchain_exceptions(fallback_value: Any = None, log_level: str = "error"):
    """
    LangChain 관련 예외를 처리하는 데코레이터.
    발생한 예외를 로깅하고 지정된 폴백 값을 반환합니다.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            try:
                return await func(*args, **kwargs)
            except (ValueError, ToolException, LangChainException) as e:
                getattr(logger, log_level)(f"{func.__name__} 실행 중 오류 발생 ({type(e).__name__}): {e}")
                return fallback_value
            except Exception as e:
                getattr(logger, log_level)(f"{func.__name__} 예상치 못한 오류 발생: {e}")
                return fallback_value

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except (ValueError, ToolException, LangChainException) as e:
                getattr(logger, log_level)(f"{func.__name__} 실행 중 오류 발생 ({type(e).__name__}): {e}")
                return fallback_value
            except Exception as e:
                getattr(logger, log_level)(f"{func.__name__} 예상치 못한 오류 발생: {e}")
                return fallback_value

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator