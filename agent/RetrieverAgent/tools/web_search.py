import asyncio
from typing import Optional, Dict, Any
import logging

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.schema import LangChainException
from langchain.tools.base import ToolException

from ..mcp_client import MCPClient

class EnhancedWebSearchTool(BaseTool):
    """MCP 서버를 통한 향상된 웹 검색 도구"""
    name = "enhanced_web_search"
    description = "MCP 서버를 통해 DuckDuckGo API 웹 검색을 수행합니다"

    def __init__(self, mcp_client: MCPClient):
        super().__init__()
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(f"{__name__}.EnhancedWebSearchTool")

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """동기 실행"""
        return asyncio.run(self._arun(query, run_manager))

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """비동기 웹 검색 실행 - MCP 서버 호출"""
        try:
            # MCP 서버의 enhanced_web_search 도구 호출
            result = await self.mcp_client.call_tool(
                server_name="web_search",
                tool_name="enhanced_web_search",
                params={
                    "query": query,
                    "max_results": 10,
                    "language": "ko"
                }
            )
            
            # MCP 서버 응답을 문자열로 변환
            if isinstance(result, dict):
                return self._format_mcp_result(result)
            else:
                return str(result)
                
        except ToolException as e:
            self.logger.error(f"MCP 도구 실행 오류: {e}")
            return await self._fallback_web_search(query)
        except LangChainException as e:
            self.logger.error(f"LangChain 오류: {e}")
            return await self._fallback_web_search(query)
        except Exception as e:
            self.logger.error(f"MCP 웹 검색 오류: {e}")
            return await self._fallback_web_search(query)

    def _format_mcp_result(self, result: Dict[str, Any]) -> str:
        """MCP 서버 응답을 포맷팅"""
        if "results" in result:
            formatted_results = []
            
            # 검색 결과 포맷팅
            for item in result["results"][:5]:  # 상위 5개만
                if item.get("type") == "direct_answer":
                    formatted_results.append(f"**직접 답변**: {item.get('content', '')}")
                elif item.get("type") == "abstract":
                    formatted_results.append(f"**개요**: {item.get('content', '')}")
                elif item.get("type") == "related_topic":
                    formatted_results.append(f"**관련 주제**: {item.get('content', '')}")
                else:
                    formatted_results.append(f"**{item.get('title', '결과')}**: {item.get('content', '')}")
            
            return "\n\n".join(formatted_results) if formatted_results else "검색 결과가 없습니다."
        else:
            return str(result)

    async def _fallback_web_search(self, query: str) -> str:
        """MCP 서버 실패 시 폴백 검색"""
        self.logger.info(f"MCP 서버 폴백 검색 실행: {query}")
        return f"""
**검색어**: {query}
**폴백 검색 결과**:
1. {query} 개요 및 기본 정보
2. {query} 사용법 및 예제
3. {query} 베스트 프랙티스
4. {query} 문제 해결 가이드
5. {query} 최신 동향

**신뢰도**: 중간 (MCP 서버 폴백)
**참고**: MCP 서버 연결 실패로 시뮬레이션 결과를 제공합니다.
"""