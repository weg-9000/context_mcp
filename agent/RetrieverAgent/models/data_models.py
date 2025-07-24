import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .enums import DataSource, ProcessingStatus, QualityLevel

@dataclass
class RetrievalTask:
    """검색 작업 정의 - 실제 사용을 위해 개선"""
    task_id: str
    query: str
    data_sources: List[DataSource]
    priority: int
    max_results: int
    freshness_requirement: float  # 0.0 ~ 1.0
    relevance_threshold: float
    language_preference: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "task_id": self.task_id,
            "query": self.query,
            "data_sources": [ds.value for ds in self.data_sources],
            "priority": self.priority,
            "max_results": self.max_results,
            "freshness_requirement": self.freshness_requirement,
            "relevance_threshold": self.relevance_threshold,
            "language_preference": self.language_preference,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at
        }

@dataclass
class RetrievedItem:
    """검색된 항목 - 실제 사용을 위해 개선"""
    item_id: str
    source: DataSource
    content: Any
    title: Optional[str] = None
    url: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    relevance_score: float = 0.0
    freshness_score: float = 0.0
    quality_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "item_id": self.item_id,
            "source": self.source.value,
            "content": str(self.content),
            "title": self.title,
            "url": self.url,
            "timestamp": self.timestamp,
            "relevance_score": self.relevance_score,
            "freshness_score": self.freshness_score,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level.value,
            "metadata": self.metadata
        }

    def calculate_overall_score(self) -> float:
        """전체 점수 계산"""
        return (
            self.relevance_score * 0.4 +
            self.freshness_score * 0.3 +
            self.quality_score * 0.3
        )

@dataclass
class SearchMetrics:
    """검색 메트릭"""
    total_searches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failed_searches: int = 0
    average_response_time: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        return (self.total_searches - self.failed_searches) / self.total_searches if self.total_searches > 0 else 0.0