import json
import time
import hashlib
from typing import Dict, Optional, Any
import logging

class SearchCache:
    """검색 결과 캐싱 시스템 - 개선된 버전"""

    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl
        self._max_size = max_size
        self._access_times: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.SearchCache")

    def _get_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """캐시 키 생성"""
        cache_data = f"{query}_{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def get(self, query: str, params: Optional[Dict] = None) -> Optional[str]:
        """캐시에서 결과 조회"""
        key = self._get_cache_key(query, params)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._access_times[key] = time.time()
                self.logger.debug(f"캐시 히트: {query[:50]}...")
                return result
            else:
                self._remove_cache_entry(key)
                self.logger.debug(f"캐시 만료: {query[:50]}...")
        return None

    def set(self, query: str, result: str, params: Optional[Dict] = None):
        """캐시에 결과 저장"""
        if len(self._cache) >= self._max_size:
            self._evict_oldest()

        key = self._get_cache_key(query, params)
        self._cache[key] = (result, time.time())
        self._access_times[key] = time.time()
        self.logger.debug(f"캐시 저장: {query[:50]}...")

    def _remove_cache_entry(self, key: str):
        """캐시 엔트리 제거"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]

    def _evict_oldest(self):
        """가장 오래된 캐시 엔트리 제거 (LRU)"""
        if not self._access_times:
            return
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_cache_entry(oldest_key)
        self.logger.debug("LRU 캐시 제거 수행")

    def clear_expired(self):
        """만료된 캐시 정리"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._ttl
        ]
        for key in expired_keys:
            self._remove_cache_entry(key)

        if expired_keys:
            self.logger.info(f"만료된 캐시 {len(expired_keys)}개 정리 완료")

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        return {
            "total_entries": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "memory_usage_mb": len(str(self._cache)) / (1024 * 1024),
            "utilization_rate": len(self._cache) / self._max_size
        }

    def clear_all(self):
        """모든 캐시 제거"""
        self._cache.clear()
        self._access_times.clear()
        self.logger.info("모든 캐시 제거 완료")