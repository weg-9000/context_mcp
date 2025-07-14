import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import quote
from datetime import datetime

# HTTP 클라이언트
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# MCP 서버 기본 클래스 (가상의 MCP 라이브러리)
try:
    from mcp.server import MCPServer
    from mcp.types import Tool, Resource, TextContent, ImageContent
    MCP_AVAILABLE = True
except ImportError:
    # MCP 라이브러리가 없는 경우 기본 구현
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

# 캐싱 시스템 (기존 SearchCache와 동일)
class WebSearchCache:
    """웹 검색 결과 캐싱"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 500):
        self._cache = {}
        self._ttl = ttl
        self._max_size = max_size
        self._access_times = {}
    
    def _get_cache_key(self, query: str, params: Dict = None) -> str:
        cache_data = f"{query}_{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def get(self, query: str, params: Dict = None) -> Optional[str]:
        key = self._get_cache_key(query, params)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._access_times[key] = time.time()
                return result
            else:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
        return None
    
    def set(self, query: str, result: str, params: Dict = None):
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        key = self._get_cache_key(query, params)
        self._cache[key] = (result, time.time())
        self._access_times[key] = time.time()
    
    def _evict_oldest(self):
        if not self._access_times:
            return
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        if oldest_key in self._cache:
            del self._cache[oldest_key]
        if oldest_key in self._access_times:
            del self._access_times[oldest_key]


class WebSearchMCPServer(MCPServer):
    """웹 검색 전용 MCP 서버"""
    
    def __init__(self):
        super().__init__("web-search-server")
        
        # HTTP 세션 및 캐시 초기화
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.cache = WebSearchCache()
        
        # 로깅 설정
        self.logger = logging.getLogger(f"{__name__}.WebSearchMCPServer")
        
        # 성능 메트릭
        self.metrics = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failed_searches": 0,
            "average_response_time": 0.0
        }
        
        # 도구 및 리소스 등록
        self.register_tool(self.enhanced_web_search)
        self.register_tool(self.search_with_filters)
        self.register_tool(self.get_search_suggestions)
        
        self.register_resource(self.get_search_cache)
        self.register_resource(self.get_search_metrics)
    
    async def initialize(self):
        """서버 초기화"""
        if AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=30)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info("웹 검색 MCP 서버 초기화 완료")
        else:
            self.logger.warning("aiohttp 사용 불가, 제한된 기능으로 동작")
    
    async def enhanced_web_search(self, query: str, max_results: int = 10, language: str = "ko") -> dict:
        """향상된 웹 검색 (기존 EnhancedWebSearchTool과 동일한 기능)"""
        start_time = time.time()
        self.metrics["total_searches"] += 1
        
        # 캐시 확인
        cache_params = {"max_results": max_results, "language": language}
        cached_result = self.cache.get(query, cache_params)
        
        if cached_result:
            self.metrics["cache_hits"] += 1
            self.logger.info(f"캐시 히트: {query}")
            return json.loads(cached_result)
        
        self.metrics["cache_misses"] += 1
        
        try:
            if not self.http_session:
                return await self._fallback_web_search(query, max_results)
            
            # DuckDuckGo API 호출 (기존과 동일)
            search_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
            
            async with self.http_session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = await self._parse_search_results(data, query, max_results)
                    
                    # 결과 캐싱
                    self.cache.set(query, json.dumps(results), cache_params)
                    
                    # 메트릭 업데이트
                    processing_time = time.time() - start_time
                    self._update_average_response_time(processing_time)
                    
                    return results
                else:
                    return await self._fallback_web_search(query, max_results)
        
        except Exception as e:
            self.metrics["failed_searches"] += 1
            self.logger.error(f"웹 검색 실패: {query}, 오류: {e}")
            return await self._fallback_web_search(query, max_results)
    
    async def _parse_search_results(self, data: Dict[str, Any], query: str, max_results: int) -> dict:
        """검색 결과 파싱 (기존 로직 유지)"""
        results = {
            "query": query,
            "search_engine": "DuckDuckGo",
            "timestamp": datetime.now().isoformat(),
            "results": [],
            "summary": {},
            "metadata": {
                "total_found": 0,
                "processing_time": time.time(),
                "language": "ko",
                "source": "real_api"
            }
        }
        
        parsed_results = []
        
        # 직접 답변
        if data.get("Answer"):
            parsed_results.append({
                "type": "direct_answer",
                "title": "직접 답변",
                "content": data["Answer"],
                "relevance_score": 1.0,
                "source": "duckduckgo_instant"
            })
        
        # 개요
        if data.get("Abstract"):
            parsed_results.append({
                "type": "abstract",
                "title": "개요",
                "content": data["Abstract"],
                "url": data.get("AbstractURL", ""),
                "source": data.get("AbstractSource", ""),
                "relevance_score": 0.95
            })
        
        # 관련 주제
        if data.get("RelatedTopics"):
            for i, topic in enumerate(data["RelatedTopics"][:max_results-len(parsed_results)]):
                if isinstance(topic, dict) and topic.get("Text"):
                    text = topic["Text"]
                    if len(text) > 200:
                        text = text[:200] + "..."
                    
                    parsed_results.append({
                        "type": "related_topic",
                        "title": f"관련 주제 {i+1}",
                        "content": text,
                        "url": topic.get("FirstURL", ""),
                        "relevance_score": 0.9 - (i * 0.1)
                    })
        
        # 결과가 부족한 경우 시뮬레이션 결과 추가
        if len(parsed_results) < max_results:
            simulation_results = await self._generate_simulation_results(query, max_results - len(parsed_results))
            parsed_results.extend(simulation_results)
        
        results["results"] = parsed_results[:max_results]
        results["metadata"]["total_found"] = len(results["results"])
        
        # 요약 정보
        results["summary"] = {
            "direct_answers": len([r for r in results["results"] if r["type"] == "direct_answer"]),
            "abstracts": len([r for r in results["results"] if r["type"] == "abstract"]),
            "related_topics": len([r for r in results["results"] if r["type"] == "related_topic"]),
            "simulated": len([r for r in results["results"] if r.get("source") == "simulation"]),
            "average_relevance": sum(r["relevance_score"] for r in results["results"]) / len(results["results"]) if results["results"] else 0
        }
        
        return results
    
    async def _generate_simulation_results(self, query: str, count: int) -> List[dict]:
        """시뮬레이션 결과 생성 (기존 폴백 로직과 동일)"""
        simulation_templates = [
            "완벽 가이드", "실무 예제", "베스트 프랙티스", 
            "문제 해결", "최신 동향", "심화 학습", "기초 개념"
        ]
        
        results = []
        for i in range(count):
            template = simulation_templates[i % len(simulation_templates)]
            relevance = 0.8 - (i * 0.05)
            
            results.append({
                "type": "simulated",
                "title": f"{query} - {template}",
                "content": f"{query}에 대한 {template.lower()} 정보를 제공합니다. "
                          f"{'기초부터 고급까지' if i == 0 else '실제 사용 사례와' if i == 1 else '전문가의 조언과'} "
                          f"함께 설명합니다.",
                "url": f"https://example.com/{query.replace(' ', '-').lower()}-{template.lower()}",
                "relevance_score": round(relevance, 2),
                "source": "simulation",
                "domain": "example.com",
                "last_updated": datetime.now().strftime('%Y-%m-%d')
            })
        
        return results
    
    async def _fallback_web_search(self, query: str, max_results: int) -> dict:
        """폴백 웹 검색 (기존과 동일)"""
        self.logger.info(f"폴백 검색 실행: {query}")
        
        results = {
            "query": query,
            "search_engine": "Fallback Simulation",
            "timestamp": datetime.now().isoformat(),
            "results": await self._generate_simulation_results(query, max_results),
            "metadata": {
                "total_found": max_results,
                "processing_time": time.time(),
                "language": "ko",
                "source": "fallback_simulation"
            }
        }
        
        # 요약 정보
        results["summary"] = {
            "simulated": len(results["results"]),
            "average_relevance": sum(r["relevance_score"] for r in results["results"]) / len(results["results"]),
            "note": "실제 API 호출 실패로 시뮬레이션 결과 제공"
        }
        
        return results
    
    async def search_with_filters(self, query: str, filters: dict = None) -> dict:
        """필터링된 검색"""
        filters = filters or {}
        
        # 기본 검색 수행
        base_results = await self.enhanced_web_search(
            query, 
            max_results=filters.get("max_results", 10),
            language=filters.get("language", "ko")
        )
        
        # 필터 적용
        filtered_results = []
        for result in base_results["results"]:
            # 관련성 필터
            if filters.get("min_relevance", 0) <= result["relevance_score"]:
                # 타입 필터
                if not filters.get("result_types") or result["type"] in filters["result_types"]:
                    # 도메인 필터
                    if not filters.get("exclude_domains") or result.get("domain") not in filters["exclude_domains"]:
                        filtered_results.append(result)
        
        base_results["results"] = filtered_results
        base_results["metadata"]["filtered"] = True
        base_results["metadata"]["applied_filters"] = filters
        
        return base_results
    
    async def get_search_suggestions(self, partial_query: str) -> dict:
        """검색 제안"""
        suggestions = [
            f"{partial_query} 사용법",
            f"{partial_query} 예제",
            f"{partial_query} 튜토리얼",
            f"{partial_query} 문제 해결",
            f"{partial_query} 최신 버전"
        ]
        
        return {
            "partial_query": partial_query,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_search_cache(self, cache_key: str = None) -> dict:
        """캐시 리소스 조회"""
        if cache_key:
            # 특정 캐시 엔트리 조회
            return {"cache_key": cache_key, "exists": cache_key in self.cache._cache}
        else:
            # 캐시 통계
            return {
                "total_entries": len(self.cache._cache),
                "max_size": self.cache._max_size,
                "ttl_seconds": self.cache._ttl,
                "hit_rate": (
                    self.metrics["cache_hits"] / 
                    (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                    if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 else 0
                )
            }
    
    async def get_search_metrics(self) -> dict:
        """검색 메트릭 리소스"""
        return {
            "server_name": "web-search-server",
            "metrics": self.metrics.copy(),
            "uptime": time.time(),
            "status": "active"
        }
    
    def _update_average_response_time(self, current_time: float):
        """평균 응답 시간 업데이트"""
        total = self.metrics["total_searches"]
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
        self.logger.info("웹 검색 MCP 서버 정리 완료")


# 서버 실행 함수
async def main():
    """웹 검색 MCP 서버 실행"""
    server = WebSearchMCPServer()
    await server.initialize()
    
    # MCP 프로토콜 서버 시작 (실제 구현에서는 MCP 라이브러리 사용)
    print("웹 검색 MCP 서버가 시작되었습니다.")
    print("사용 가능한 도구:")
    print("- enhanced_web_search: 향상된 웹 검색")
    print("- search_with_filters: 필터링된 검색")
    print("- get_search_suggestions: 검색 제안")
    
    try:
        # 서버 실행 대기
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await server.cleanup()
        print("웹 검색 MCP 서버가 종료되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())