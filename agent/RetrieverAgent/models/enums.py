from enum import Enum

class RetrievalMode(Enum):
    """검색 모드"""
    COMPREHENSIVE = "comprehensive"  # 포괄적 검색
    FOCUSED = "focused"              # 집중 검색
    REAL_TIME = "real_time"          # 실시간 검색
    MULTIMODAL = "multimodal"        # 멀티모달 검색

class DataSource(Enum):
    """실제 사용되는 데이터 소스만 유지"""
    WEB_SEARCH = "web_search"
    API_ENDPOINT = "api_endpoint"
    MULTIMEDIA = "multimedia"

    # 향후 확장 가능한 소스들 (현재는 주석 처리)
    # DATABASE = "database"
    # FILE_SYSTEM = "file_system"
    # SOCIAL_MEDIA = "social_media"
    # NEWS_FEED = "news_feed"
    # ACADEMIC = "academic"

class ProcessingStatus(Enum):
    """처리 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class QualityLevel(Enum):
    """품질 수준"""
    EXCELLENT = "excellent"
    GOOD = "good"
    MEDIUM = "medium"
    BASIC = "basic"
    POOR = "poor"