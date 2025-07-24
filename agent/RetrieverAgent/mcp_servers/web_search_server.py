import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
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
logger = logging.getLogger("web-search-server")

class WebSearchServer:
    """웹 검색 MCP 서버"""
    
    def __init__(self):
        self.server = Server("web-search")
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize_session(self):
        """HTTP 세션 초기화"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
    
    async def search_google(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Google 검색 수행"""
        await self.initialize_session()
        
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={min(limit, 10)}"
        
        try:
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_google_results(html)
                else:
                    logger.error(f"Google 검색 실패: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Google 검색 중 오류: {e}")
            return []
    
    def _parse_google_results(self, html: str) -> List[Dict[str, Any]]:
        """Google 검색 결과 파싱"""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Google 검색 결과 컨테이너 찾기
        search_results = soup.find_all('div', class_='g')
        
        for result in search_results:
            try:
                # 제목 추출
                title_elem = result.find('h3')
                title = title_elem.get_text() if title_elem else "제목 없음"
                
                # URL 추출
                link_elem = result.find('a')
                url = link_elem.get('href') if link_elem else ""
                
                # 설명 추출
                desc_elem = result.find('span', class_=['aCOpRe', 'st'])
                if not desc_elem:
                    desc_elem = result.find('div', class_=['VwiC3b', 's3v9rd'])
                description = desc_elem.get_text() if desc_elem else "설명 없음"
                
                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "description": description[:200] + "..." if len(description) > 200 else description
                    })
                    
            except Exception as e:
                logger.warning(f"검색 결과 파싱 중 오류: {e}")
                continue
        
        return results[:10]  # 최대 10개 결과
    
    def setup_handlers(self):
        """MCP 핸들러 설정"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """사용 가능한 도구 목록 반환"""
            return [
                Tool(
                    name="enhanced_web_search",
                    description="Google을 통한 웹 검색을 수행합니다",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색할 쿼리"
                            },
                            "limit": {
                                "type": "number",
                                "description": "반환할 결과 수 (기본값: 5, 최대: 10)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "language": {
                                "type": "string",
                                "description": "검색 언어 (기본값: auto)",
                                "default": "auto"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """도구 호출 처리"""
            if name == "enhanced_web_search":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 5)
                
                if not query:
                    return CallToolResult(
                        content=[TextContent(type="text", text="검색 쿼리가 필요합니다.")]
                    )
                
                try:
                    results = await self.search_google(query, limit)
                    
                    if results:
                        # 결과를 구조화된 형태로 반환
                        response_data = {
                            "query": query,
                            "results": results,
                            "total_found": len(results),
                            "status": "success"
                        }
                        
                        # 텍스트 형태로도 포맷팅
                        formatted_text = f"**검색어**: {query}\n\n"
                        for i, result in enumerate(results, 1):
                            formatted_text += f"**{i}. {result['title']}**\n"
                            formatted_text += f"URL: {result['url']}\n"
                            formatted_text += f"설명: {result['description']}\n\n"
                        
                        return CallToolResult(
                            content=[
                                TextContent(type="text", text=formatted_text),
                                TextContent(type="text", text=json.dumps(response_data, ensure_ascii=False, indent=2))
                            ]
                        )
                    else:
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"'{query}'에 대한 검색 결과를 찾을 수 없습니다.")]
                        )
                        
                except Exception as e:
                    logger.error(f"웹 검색 중 오류: {e}")
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"검색 중 오류가 발생했습니다: {str(e)}")]
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
    web_server = WebSearchServer()
    web_server.setup_handlers()
    
    try:
        # STDIO를 통한 MCP 서버 실행
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await web_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    protocolVersion="2024-11-05",
                    capabilities=web_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    finally:
        await web_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
