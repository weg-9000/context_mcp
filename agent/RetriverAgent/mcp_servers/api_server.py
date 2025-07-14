import asyncio
import json
import sys
import os
from typing import Any, Dict, List, Optional
import aiohttp
import logging

try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import Tool, TextContent, CallToolResult
    import mcp.types as types
except ImportError:
    print("MCP 라이브러리가 설치되지 않았습니다. pip install mcp 를 실행하세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-search-server")

class APISearchServer:
    """API 검색 MCP 서버"""
    
    def __init__(self):
        self.server = Server("api-search")
        self.session: Optional[aiohttp.ClientSession] = None
        self.github_token = os.getenv("GITHUB_TOKEN")
        
    async def initialize_session(self):
        """HTTP 세션 초기화"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'RetrieverAgent-MCP-Server/1.0'
            }
            
            # GitHub 토큰이 있으면 헤더에 추가
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
                
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
    
    async def search_github(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """GitHub 리포지토리 검색"""
        await self.initialize_session()
        
        search_url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, 10)
        }
        
        try:
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "items": [
                            {
                                "name": item["name"],
                                "full_name": item["full_name"],
                                "description": item.get("description", "설명 없음"),
                                "url": item["html_url"],
                                "stars": item["stargazers_count"],
                                "language": item.get("language", "Unknown"),
                                "updated_at": item["updated_at"]
                            }
                            for item in data.get("items", [])
                        ],
                        "total_count": data.get("total_count", 0)
                    }
                else:
                    logger.error(f"GitHub API 오류: HTTP {response.status}")
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"GitHub 검색 중 오류: {e}")
            return {"status": "error", "message": str(e)}
    
    async def search_stackoverflow(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Stack Overflow 질문 검색"""
        await self.initialize_session()
        
        search_url = "https://api.stackexchange.com/2.3/search/advanced"
        params = {
            "q": query,
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "pagesize": min(limit, 10)
        }
        
        try:
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "items": [
                            {
                                "title": item["title"],
                                "url": item["link"],
                                "score": item["score"],
                                "answer_count": item["answer_count"],
                                "view_count": item["view_count"],
                                "is_answered": item["is_answered"],
                                "creation_date": item["creation_date"],
                                "tags": item.get("tags", [])
                            }
                            for item in data.get("items", [])
                        ],
                        "total": data.get("total", 0)
                    }
                else:
                    logger.error(f"Stack Overflow API 오류: HTTP {response.status}")
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"Stack Overflow 검색 중 오류: {e}")
            return {"status": "error", "message": str(e)}
    
    def setup_handlers(self):
        """MCP 핸들러 설정"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """사용 가능한 도구 목록 반환"""
            return [
                Tool(
                    name="api_search",
                    description="GitHub 리포지토리와 Stack Overflow 질문을 검색합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색할 쿼리"
                            },
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["github", "stackoverflow"]
                                },
                                "description": "검색할 소스 (기본값: ['github', 'stackoverflow'])",
                                "default": ["github", "stackoverflow"]
                            },
                            "limit": {
                                "type": "number",
                                "description": "각 소스별 결과 수 (기본값: 5)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """도구 호출 처리"""
            if name == "api_search":
                query = arguments.get("query", "")
                sources = arguments.get("sources", ["github", "stackoverflow"])
                limit = arguments.get("limit", 5)
                
                if not query:
                    return CallToolResult(
                        content=[TextContent(type="text", text="검색 쿼리가 필요합니다.")]
                    )
                
                results = {}
                
                # GitHub 검색
                if "github" in sources:
                    try:
                        github_result = await self.search_github(query, limit)
                        results["github"] = github_result
                    except Exception as e:
                        logger.error(f"GitHub 검색 오류: {e}")
                        results["github"] = {"status": "error", "message": str(e)}
                
                # Stack Overflow 검색
                if "stackoverflow" in sources:
                    try:
                        so_result = await self.search_stackoverflow(query, limit)
                        results["stackoverflow"] = so_result
                    except Exception as e:
                        logger.error(f"Stack Overflow 검색 오류: {e}")
                        results["stackoverflow"] = {"status": "error", "message": str(e)}
                
                # 결과 포맷팅
                formatted_text = f"**검색어**: {query}\n\n"
                
                # GitHub 결과 포맷팅
                if "github" in results:
                    github_data = results["github"]
                    if github_data.get("status") == "success":
                        formatted_text += f"**GitHub API**: 관련 리포지토리 {len(github_data['items'])}개 발견\n"
                        for item in github_data["items"][:3]:
                            formatted_text += f"- **{item['name']}**: {item['description']}\n"
                            formatted_text += f"  ⭐ {item['stars']} | 🔗 {item['url']}\n"
                    else:
                        formatted_text += f"**GitHub API**: 오류 - {github_data.get('message', '알 수 없는 오류')}\n"
                    formatted_text += "\n"
                
                # Stack Overflow 결과 포맷팅
                if "stackoverflow" in results:
                    so_data = results["stackoverflow"]
                    if so_data.get("status") == "success":
                        formatted_text += f"**Stack Overflow API**: 관련 질문 {len(so_data['items'])}개 발견\n"
                        for item in so_data["items"][:3]:
                            formatted_text += f"- **{item['title']}**\n"
                            formatted_text += f"  👍 {item['score']} | 💬 {item['answer_count']} | 🔗 {item['url']}\n"
                    else:
                        formatted_text += f"**Stack Overflow API**: 오류 - {so_data.get('message', '알 수 없는 오류')}\n"
                
                # 요약 정보
                successful_sources = sum(1 for result in results.values() if result.get("status") == "success")
                formatted_text += f"\n**데이터 품질**: {'높음' if successful_sources > 0 else '낮음'} (API 기반 구조화된 데이터)\n"
                formatted_text += f"**총 소스**: {len(results)}개\n"
                formatted_text += f"**성공한 소스**: {successful_sources}개\n"
                
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=formatted_text),
                        TextContent(type="text", text=json.dumps({
                            "query": query,
                            "results": results,
                            "summary": {
                                "total_sources": len(results),
                                "successful_sources": successful_sources
                            }
                        }, ensure_ascii=False, indent=2))
                    ]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"알 수 없는 도구: {name}")]
                )
    
    async def cleanup(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()

async def main():
    """메인 실행 함수"""
    api_server = APISearchServer()
    api_server.setup_handlers()
    
    try:
        # STDIO를 통한 MCP 서버 실행
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await api_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    protocolVersion="2024-11-05",
                    capabilities=api_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    finally:
        await api_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())