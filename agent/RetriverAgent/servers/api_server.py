import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote

# HTTP 클라이언트
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# MCP 서버 기본 클래스
try:
    from mcp.server import MCPServer
    from mcp.types import Tool, Resource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    
    class MCPServer:
        def __init__(self, name: str):
            self.name = name
            self.tools = {}
            self.resources = {}
        
        def register_tool(self, func):
            self.tools[func.__name__] = func
        
        def register_resource(self, func):
            self.resources[func.__name__] = func


class APIMCPServer(MCPServer):
    """API 데이터 수집 전용 MCP 서버"""
    
    def __init__(self):
        super().__init__("api-server")
        
        # HTTP 세션
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # API 설정
        self.api_configs = {
            "github": {
                "base_url": "https://api.github.com",
                "rate_limit": 60,  # requests per hour
                "headers": {"Accept": "application/vnd.github.v3+json"}
            },
            "stackoverflow": {
                "base_url": "https://api.stackexchange.com/2.3",
                "rate_limit": 300,  # requests per day
                "headers": {"Accept": "application/json"}
            }
        }
        
        # 로깅 설정
        self.logger = logging.getLogger(f"{__name__}.APIMCPServer")
        
        # 성능 메트릭
        self.metrics = {
            "total_requests": 0,
            "github_requests": 0,
            "stackoverflow_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        
        # 도구 및 리소스 등록
        self.register_tool(self.api_search)
        self.register_tool(self.github_search)
        self.register_tool(self.stackoverflow_search)
        self.register_tool(self.documentation_search)
        
        self.register_resource(self.get_api_metrics)
        self.register_resource(self.get_rate_limits)
    
    async def initialize(self):
        """서버 초기화"""
        if AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=30)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info("API MCP 서버 초기화 완료")
        else:
            self.logger.warning("aiohttp 사용 불가, 시뮬레이션 모드로 동작")
    
    async def api_search(self, query: str, sources: List[str] = None) -> dict:
        """통합 API 검색 (기존 APIRetrieverTool과 동일한 기능)"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        if sources is None:
            sources = ["github", "stackoverflow"]
        
        results = {
            "query": query,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "results": {},
            "summary": {},
            "metadata": {
                "total_sources": len(sources),
                "processing_time": 0,
                "api_calls_made": 0
            }
        }
        
        try:
            # 각 소스별 검색 수행
            for source in sources:
                try:
                    if source == "github":
                        source_results = await self.github_search(query)
                        results["results"]["github"] = source_results
                        self.metrics["github_requests"] += 1
                    
                    elif source == "stackoverflow":
                        source_results = await self.stackoverflow_search(query)
                        results["results"]["stackoverflow"] = source_results
                        self.metrics["stackoverflow_requests"] += 1
                    
                    else:
                        # 알려지지 않은 소스는 시뮬레이션
                        source_results = await self._simulate_api_source(query, source)
                        results["results"][source] = source_results
                    
                    results["metadata"]["api_calls_made"] += 1
                    
                except Exception as e:
                    self.logger.error(f"{source} API 검색 실패: {e}")
                    results["results"][source] = {
                        "error": str(e),
                        "status": "failed",
                        "items": []
                    }
            
            # 요약 정보 생성
            results["summary"] = await self._generate_api_summary(results["results"])
            
            # 메트릭 업데이트
            processing_time = time.time() - start_time
            results["metadata"]["processing_time"] = round(processing_time, 3)
            self._update_average_response_time(processing_time)
            
            return results
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"API 검색 실패: {query}, 오류: {e}")
            return await self._fallback_api_search(query, sources)
    
    async def github_search(self, query: str, search_type: str = "repositories") -> dict:
        """GitHub API 검색"""
        if not self.http_session:
            return await self._simulate_github_search(query)
        
        try:
            # GitHub API 엔드포인트
            if search_type == "repositories":
                url = f"{self.api_configs['github']['base_url']}/search/repositories"
                params = {"q": query, "sort": "stars", "order": "desc", "per_page": 5}
            else:
                url = f"{self.api_configs['github']['base_url']}/search/code"
                params = {"q": query, "per_page": 5}
            
            headers = self.api_configs['github']['headers'].copy()
            
            async with self.http_session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_github_results(data, query, search_type)
                else:
                    self.logger.warning(f"GitHub API 오류: {response.status}")
                    return await self._simulate_github_search(query)
        
        except Exception as e:
            self.logger.error(f"GitHub 검색 실패: {e}")
            return await self._simulate_github_search(query)
    
    async def _parse_github_results(self, data: dict, query: str, search_type: str) -> dict:
        """GitHub 검색 결과 파싱"""
        items = []
        
        for item in data.get("items", [])[:5]:
            if search_type == "repositories":
                parsed_item = {
                    "type": "repository",
                    "name": item.get("name", ""),
                    "full_name": item.get("full_name", ""),
                    "description": item.get("description", ""),
                    "url": item.get("html_url", ""),
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "language": item.get("language", ""),
                    "updated_at": item.get("updated_at", ""),
                    "topics": item.get("topics", [])
                }
            else:
                parsed_item = {
                    "type": "code",
                    "name": item.get("name", ""),
                    "path": item.get("path", ""),
                    "repository": item.get("repository", {}).get("full_name", ""),
                    "url": item.get("html_url", ""),
                    "score": item.get("score", 0)
                }
            
            items.append(parsed_item)
        
        return {
            "source": "github",
            "search_type": search_type,
            "query": query,
            "total_count": data.get("total_count", 0),
            "items": items,
            "status": "success"
        }
    
    async def _simulate_github_search(self, query: str) -> dict:
        """GitHub 검색 시뮬레이션"""
        simulated_repos = [
            {
                "type": "repository",
                "name": f"{query}-awesome",
                "full_name": f"developer/{query}-awesome",
                "description": f"Awesome {query} resources and tools",
                "url": f"https://github.com/developer/{query}-awesome",
                "stars": 1250,
                "forks": 180,
                "language": "Python",
                "updated_at": datetime.now().isoformat(),
                "topics": [query.lower(), "awesome", "resources"]
            },
            {
                "type": "repository", 
                "name": f"{query}-tutorial",
                "full_name": f"tutorial/{query}-tutorial",
                "description": f"Complete {query} tutorial with examples",
                "url": f"https://github.com/tutorial/{query}-tutorial",
                "stars": 890,
                "forks": 120,
                "language": "JavaScript",
                "updated_at": (datetime.now() - timedelta(days=5)).isoformat(),
                "topics": [query.lower(), "tutorial", "examples"]
            },
            {
                "type": "repository",
                "name": f"{query}-framework",
                "full_name": f"framework/{query}-framework",
                "description": f"Modern {query} framework for rapid development",
                "url": f"https://github.com/framework/{query}-framework",
                "stars": 2340,
                "forks": 310,
                "language": "TypeScript",
                "updated_at": (datetime.now() - timedelta(days=2)).isoformat(),
                "topics": [query.lower(), "framework", "development"]
            }
        ]
        
        return {
            "source": "github",
            "search_type": "repositories",
            "query": query,
            "total_count": len(simulated_repos),
            "items": simulated_repos,
            "status": "simulated"
        }
    
    async def stackoverflow_search(self, query: str) -> dict:
        """Stack Overflow API 검색"""
        if not self.http_session:
            return await self._simulate_stackoverflow_search(query)
        
        try:
            url = f"{self.api_configs['stackoverflow']['base_url']}/search/advanced"
            params = {
                "q": query,
                "site": "stackoverflow",
                "sort": "votes",
                "order": "desc",
                "pagesize": 5
            }
            
            headers = self.api_configs['stackoverflow']['headers'].copy()
            
            async with self.http_session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_stackoverflow_results(data, query)
                else:
                    self.logger.warning(f"Stack Overflow API 오류: {response.status}")
                    return await self._simulate_stackoverflow_search(query)
        
        except Exception as e:
            self.logger.error(f"Stack Overflow 검색 실패: {e}")
            return await self._simulate_stackoverflow_search(query)
    
    async def _parse_stackoverflow_results(self, data: dict, query: str) -> dict:
        """Stack Overflow 검색 결과 파싱"""
        items = []
        
        for item in data.get("items", [])[:5]:
            parsed_item = {
                "type": "question",
                "title": item.get("title", ""),
                "question_id": item.get("question_id", 0),
                "url": item.get("link", ""),
                "score": item.get("score", 0),
                "answer_count": item.get("answer_count", 0),
                "view_count": item.get("view_count", 0),
                "tags": item.get("tags", []),
                "is_answered": item.get("is_answered", False),
                "creation_date": item.get("creation_date", 0),
                "last_activity_date": item.get("last_activity_date", 0)
            }
            items.append(parsed_item)
        
        return {
            "source": "stackoverflow",
            "query": query,
            "total_count": len(items),
            "items": items,
            "status": "success"
        }
    
    async def _simulate_stackoverflow_search(self, query: str) -> dict:
        """Stack Overflow 검색 시뮬레이션"""
        simulated_questions = [
            {
                "type": "question",
                "title": f"How to implement {query} in Python?",
                "question_id": 12345678,
                "url": f"https://stackoverflow.com/questions/12345678/how-to-implement-{query.lower().replace(' ', '-')}-in-python",
                "score": 45,
                "answer_count": 3,
                "view_count": 1250,
                "tags": [query.lower(), "python", "implementation"],
                "is_answered": True,
                "creation_date": int((datetime.now() - timedelta(days=30)).timestamp()),
                "last_activity_date": int((datetime.now() - timedelta(days=5)).timestamp())
            },
            {
                "type": "question",
                "title": f"Best practices for {query}",
                "question_id": 87654321,
                "url": f"https://stackoverflow.com/questions/87654321/best-practices-for-{query.lower().replace(' ', '-')}",
                "score": 67,
                "answer_count": 5,
                "view_count": 2100,
                "tags": [query.lower(), "best-practices", "design-patterns"],
                "is_answered": True,
                "creation_date": int((datetime.now() - timedelta(days=60)).timestamp()),
                "last_activity_date": int((datetime.now() - timedelta(days=10)).timestamp())
            },
            {
                "type": "question",
                "title": f"Common errors when using {query}",
                "question_id": 11223344,
                "url": f"https://stackoverflow.com/questions/11223344/common-errors-when-using-{query.lower().replace(' ', '-')}",
                "score": 23,
                "answer_count": 2,
                "view_count": 890,
                "tags": [query.lower(), "debugging", "troubleshooting"],
                "is_answered": True,
                "creation_date": int((datetime.now() - timedelta(days=15)).timestamp()),
                "last_activity_date": int((datetime.now() - timedelta(days=3)).timestamp())
            }
        ]
        
        return {
            "source": "stackoverflow",
            "query": query,
            "total_count": len(simulated_questions),
            "items": simulated_questions,
            "status": "simulated"
        }
    
    async def documentation_search(self, query: str, doc_sources: List[str] = None) -> dict:
        """문서 검색"""
        if doc_sources is None:
            doc_sources = ["official", "community", "tutorials"]
        
        results = {
            "query": query,
            "doc_sources": doc_sources,
            "timestamp": datetime.now().isoformat(),
            "documents": []
        }
        
        # 시뮬레이션 문서 생성
        for i, source in enumerate(doc_sources):
            doc = {
                "source": source,
                "title": f"{query} - {source.title()} Documentation",
                "url": f"https://docs.example.com/{source}/{query.lower().replace(' ', '-')}",
                "description": f"Official {source} documentation for {query}",
                "sections": [
                    "Getting Started",
                    "API Reference", 
                    "Examples",
                    "Best Practices",
                    "Troubleshooting"
                ],
                "last_updated": (datetime.now() - timedelta(days=i*5)).isoformat(),
                "relevance_score": 0.9 - (i * 0.1)
            }
            results["documents"].append(doc)
        
        return results
    
    async def _simulate_api_source(self, query: str, source: str) -> dict:
        """알려지지 않은 API 소스 시뮬레이션"""
        return {
            "source": source,
            "query": query,
            "items": [
                {
                    "type": "generic",
                    "title": f"{query} resource from {source}",
                    "description": f"Relevant {query} information from {source} API",
                    "url": f"https://{source}.com/search?q={query}",
                    "score": 0.8
                }
            ],
            "status": "simulated"
        }
    
    async def _generate_api_summary(self, results: dict) -> dict:
        """API 검색 결과 요약 생성"""
        summary = {
            "total_sources": len(results),
            "successful_sources": 0,
            "failed_sources": 0,
            "total_items": 0,
            "by_source": {}
        }
        
        for source, data in results.items():
            if isinstance(data, dict) and "error" not in data:
                summary["successful_sources"] += 1
                item_count = len(data.get("items", []))
                summary["total_items"] += item_count
                summary["by_source"][source] = {
                    "status": "success",
                    "items": item_count,
                    "type": data.get("search_type", "unknown")
                }
            else:
                summary["failed_sources"] += 1
                summary["by_source"][source] = {
                    "status": "failed",
                    "error": data.get("error", "Unknown error") if isinstance(data, dict) else str(data)
                }
        
        return summary
    
    async def _fallback_api_search(self, query: str, sources: List[str]) -> dict:
        """폴백 API 검색"""
        self.logger.info(f"폴백 API 검색 실행: {query}")
        
        fallback_results = {
            "query": query,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "results": {},
            "summary": {
                "note": "모든 API 호출 실패로 시뮬레이션 결과 제공"
            },
            "metadata": {
                "total_sources": len(sources),
                "processing_time": 0.1,
                "api_calls_made": 0,
                "fallback_mode": True
            }
        }
        
        # 각 소스별 시뮬레이션 결과 생성
        for source in sources:
            if source == "github":
                fallback_results["results"]["github"] = await self._simulate_github_search(query)
            elif source == "stackoverflow":
                fallback_results["results"]["stackoverflow"] = await self._simulate_stackoverflow_search(query)
            else:
                fallback_results["results"][source] = await self._simulate_api_source(query, source)
        
        fallback_results["summary"] = await self._generate_api_summary(fallback_results["results"])
        
        return fallback_results
    
    async def get_api_metrics(self) -> dict:
        """API 메트릭 리소스"""
        return {
            "server_name": "api-server",
            "metrics": self.metrics.copy(),
            "api_configs": {
                source: {
                    "base_url": config["base_url"],
                    "rate_limit": config["rate_limit"]
                }
                for source, config in self.api_configs.items()
            },
            "uptime": time.time(),
            "status": "active"
        }
    
    async def get_rate_limits(self) -> dict:
        """API 속도 제한 정보 리소스"""
        return {
            "github": {
                "limit": self.api_configs["github"]["rate_limit"],
                "remaining": "unknown",  # 실제 구현에서는 API 응답에서 추출
                "reset_time": "unknown"
            },
            "stackoverflow": {
                "limit": self.api_configs["stackoverflow"]["rate_limit"],
                "remaining": "unknown",
                "reset_time": "unknown"
            }
        }
    
    def _update_average_response_time(self, current_time: float):
        """평균 응답 시간 업데이트"""
        total = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        
        if total == 1:
            self.metrics["average_response_time"] = current_time
        else:
            new_avg = ((current_avg * (total - 1)) + current_time) / total
            self.metrics["average_response_time"] = round(new_avg, 3)
    
    async def cleanup(self):
        """리소스 정리"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        self.logger.info("API MCP 서버 정리 완료")


# 서버 실행 함수
async def main():
    """API MCP 서버 실행"""
    server = APIMCPServer()
    await server.initialize()
    
    print("API MCP 서버가 시작되었습니다.")
    print("사용 가능한 도구:")
    print("- api_search: 통합 API 검색")
    print("- github_search: GitHub 검색")
    print("- stackoverflow_search: Stack Overflow 검색")
    print("- documentation_search: 문서 검색")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await server.cleanup()
        print("API MCP 서버가 종료되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())