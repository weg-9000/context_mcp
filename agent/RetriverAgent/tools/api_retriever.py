import asyncio
from typing import Optional, Dict, Any
import logging

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from ..mcp_client import MCPClient

class APIRetrieverTool(BaseTool):
    """MCP 서버를 통한 API 검색 도구"""
    name = "api_retriever"
    description = "MCP 서버를 통해 GitHub, Stack Overflow 등의 API 데이터를 검색합니다"

    def __init__(self, mcp_client: MCPClient):
        super().__init__()
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(f"{__name__}.APIRetrieverTool")

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """동기 실행"""
        return asyncio.run(self._arun(query, run_manager))

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """비동기 API 검색 실행 - MCP 서버 호출"""
        try:
            # MCP 서버의 api_search 도구 호출
            result = await self.mcp_client.call_tool(
                server_name="api_search",
                tool_name="api_search",
                params={
                    "query": query,
                    "sources": ["github", "stackoverflow"]
                }
            )
            
            # MCP 서버 응답을 문자열로 변환
            if isinstance(result, dict):
                return self._format_api_result(result)
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"MCP API 검색 오류: {e}")
            return await self._fallback_api_search(query)

    def _format_api_result(self, result: Dict[str, Any]) -> str:
        """MCP 서버 API 응답을 포맷팅"""
        formatted_content = f"**검색어**: {result.get('query', '')}\n\n"
        
        if "results" in result:
            results = result["results"]
            
            # GitHub 결과
            if "github" in results and results["github"].get("status") == "success":
                github_items = results["github"].get("items", [])
                formatted_content += f"**GitHub API**: 관련 리포지토리 {len(github_items)}개 발견\n"
                for item in github_items[:3]:  # 상위 3개만
                    formatted_content += f"- {item.get('name', '')}: {item.get('description', '')}\n"
                formatted_content += "\n"
            
            # Stack Overflow 결과
            if "stackoverflow" in results and results["stackoverflow"].get("status") == "success":
                so_items = results["stackoverflow"].get("items", [])
                formatted_content += f"**Stack Overflow API**: 관련 질문 {len(so_items)}개 발견\n"
                for item in so_items[:3]:  # 상위 3개만
                    formatted_content += f"- {item.get('title', '')}\n"
                formatted_content += "\n"
        
        # 요약 정보
        if "summary" in result:
            summary = result["summary"]
            formatted_content += f"**데이터 품질**: 높음 (API 기반 구조화된 데이터)\n"
            formatted_content += f"**총 소스**: {summary.get('total_sources', 0)}개\n"
            formatted_content += f"**성공한 소스**: {summary.get('successful_sources', 0)}개\n"
        
        return formatted_content

    async def _fallback_api_search(self, query: str) -> str:
        """MCP 서버 실패 시 폴백 API 검색"""
        self.logger.info(f"MCP API 서버 폴백 검색 실행: {query}")
        return f"""
**검색어**: {query}
**폴백 API 검색 결과**:
- GitHub API: 관련 리포지토리 3개 발견 (시뮬레이션)
- Stack Overflow API: 관련 질문 5개 발견 (시뮬레이션)
- Documentation API: 공식 문서 2개 발견 (시뮬레이션)

**데이터 품질**: 중간 (MCP 서버 폴백)
**참고**: MCP 서버 연결 실패로 시뮬레이션 결과를 제공합니다.
"""