import asyncio
import logging
import os
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

class IsolationLevel(Enum):
    """격리 수준"""
    NONE = "none"           # 격리 없음
    BASIC = "basic"         # 기본 격리
    SECURE = "secure"       # 보안 격리
    STRICT = "strict"       # 엄격한 격리

class ResourceType(Enum):
    """리소스 타입"""
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    API_CALLS = "api_calls"

@dataclass
class ResourceLimit:
    """리소스 제한"""
    resource_type: ResourceType
    max_value: float
    current_value: float = 0.0
    warning_threshold: float = 0.8

@dataclass
class IsolationConfig:
    """격리 설정"""
    level: IsolationLevel = IsolationLevel.BASIC
    resource_limits: Dict[ResourceType, ResourceLimit] = field(default_factory=dict)
    allowed_modules: List[str] = field(default_factory=list)
    blocked_modules: List[str] = field(default_factory=list)
    network_restrictions: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    enable_logging: bool = True

class UniversalIsolationManager:
    """
    범용 격리 관리자
    
    이 클래스는 다양한 에이전트 프레임워크가 안전하게 격리된 환경에서
    실행될 수 있도록 관리하는 중앙 컨트롤러입니다.
    """
    
    def __init__(self, config: Optional[IsolationConfig] = None):
        self.config = config or IsolationConfig()
        self.logger = logging.getLogger(f"{__name__}.UniversalIsolationManager")
        
        # 활성 세션 관리
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.RLock()
        
        # 리소스 모니터링
        self.resource_monitors: Dict[str, Any] = {}
        self.monitoring_active = False
        
        # 보안 정책
        self.security_policies: Dict[str, Callable] = {}
        
        # 초기화
        self._initialize_default_limits()
        self._initialize_security_policies()
        
        self.logger.info(f"UniversalIsolationManager 초기화됨 (수준: {self.config.level.value})")
    
    def _initialize_default_limits(self):
        """기본 리소스 제한 초기화"""
        if not self.config.resource_limits:
            self.config.resource_limits = {
                ResourceType.MEMORY: ResourceLimit(ResourceType.MEMORY, 1024 * 1024 * 512),  # 512MB
                ResourceType.CPU: ResourceLimit(ResourceType.CPU, 80.0),  # 80% CPU
                ResourceType.API_CALLS: ResourceLimit(ResourceType.API_CALLS, 1000),  # 1000 calls/hour
                ResourceType.NETWORK: ResourceLimit(ResourceType.NETWORK, 100 * 1024 * 1024),  # 100MB
            }
        
        # 기본 허용/차단 모듈 설정
        if not self.config.allowed_modules:
            self.config.allowed_modules = [
                "langchain", "langchain_openai", "langchain_community",
                "crewai", "semantic_kernel", "langgraph",
                "asyncio", "aiohttp", "requests", "json", "time", "datetime",
                "logging", "typing", "dataclasses", "enum"
            ]
        
        if not self.config.blocked_modules:
            self.config.blocked_modules = [
                "subprocess", "os.system", "eval", "exec", "compile",
                "importlib", "__import__", "open", "file"
            ]
    
    def _initialize_security_policies(self):
        """보안 정책 초기화"""
        self.security_policies = {
            "module_import": self._validate_module_import,
            "resource_usage": self._validate_resource_usage,
            "network_access": self._validate_network_access,
            "file_access": self._validate_file_access,
        }
    
    async def create_isolation_session(self, session_id: str, agent_type: str = "generic") -> bool:
        """
        격리 세션 생성
        
        Args:
            session_id: 세션 식별자
            agent_type: 에이전트 타입 (예: "langchain", "crewai")
        
        Returns:
            세션 생성 성공 여부
        """
        try:
            with self.session_lock:
                if session_id in self.active_sessions:
                    self.logger.warning(f"세션 {session_id}가 이미 존재합니다.")
                    return False
                
                # 세션 정보 생성
                session_info = {
                    "agent_type": agent_type,
                    "created_at": time.time(),
                    "resource_usage": {rt: 0.0 for rt in ResourceType},
                    "security_violations": [],
                    "last_activity": time.time(),
                    "status": "active"
                }
                
                self.active_sessions[session_id] = session_info
                
                # 리소스 모니터링 시작
                if not self.monitoring_active:
                    await self._start_resource_monitoring()
                
                self.logger.info(f"격리 세션 생성됨: {session_id} (타입: {agent_type})")
                return True
                
        except Exception as e:
            self.logger.error(f"세션 생성 실패: {session_id}, 오류: {e}")
            return False
    
    async def destroy_isolation_session(self, session_id: str) -> bool:
        """
        격리 세션 제거
        
        Args:
            session_id: 세션 식별자
        
        Returns:
            세션 제거 성공 여부
        """
        try:
            with self.session_lock:
                if session_id not in self.active_sessions:
                    self.logger.warning(f"세션 {session_id}가 존재하지 않습니다.")
                    return False
                
                # 세션 정리
                session_info = self.active_sessions[session_id]
                session_info["status"] = "destroyed"
                session_info["destroyed_at"] = time.time()
                
                # 세션 제거
                del self.active_sessions[session_id]
                
                # 모든 세션이 제거되면 모니터링 중단
                if not self.active_sessions:
                    await self._stop_resource_monitoring()
                
                self.logger.info(f"격리 세션 제거됨: {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"세션 제거 실패: {session_id}, 오류: {e}")
            return False
    
    @asynccontextmanager
    async def isolation_context(self, session_id: str, agent_type: str = "generic"):
        """
        격리 컨텍스트 매니저
        
        Usage:
            async with isolation_manager.isolation_context("session_1", "langchain") as ctx:
                # 격리된 환경에서 코드 실행
                pass
        """
        success = await self.create_isolation_session(session_id, agent_type)
        if not success:
            raise RuntimeError(f"격리 세션 생성 실패: {session_id}")
        
        try:
            # 컨텍스트 정보 반환
            context_info = {
                "session_id": session_id,
                "agent_type": agent_type,
                "isolation_level": self.config.level,
                "resource_limits": self.config.resource_limits
            }
            yield context_info
        finally:
            await self.destroy_isolation_session(session_id)
    
    def validate_security_policy(self, session_id: str, policy_type: str, **kwargs) -> bool:
        """
        보안 정책 검증
        
        Args:
            session_id: 세션 식별자
            policy_type: 정책 타입 (예: "module_import", "resource_usage")
            **kwargs: 정책별 매개변수
        
        Returns:
            정책 통과 여부
        """
        try:
            if policy_type not in self.security_policies:
                self.logger.warning(f"알 수 없는 보안 정책: {policy_type}")
                return True  # 알 수 없는 정책은 허용
            
            policy_validator = self.security_policies[policy_type]
            return policy_validator(session_id, **kwargs)
            
        except Exception as e:
            self.logger.error(f"보안 정책 검증 오류: {policy_type}, {e}")
            return False
    
    def _validate_module_import(self, session_id: str, module_name: str, **kwargs) -> bool:
        """모듈 임포트 검증"""
        # 차단 모듈 확인
        for blocked in self.config.blocked_modules:
            if blocked in module_name:
                self._record_security_violation(session_id, "blocked_module_import", module_name)
                return False
        
        # 허용 모듈 확인 (STRICT 모드에서만)
        if self.config.level == IsolationLevel.STRICT:
            allowed = any(allowed_mod in module_name for allowed_mod in self.config.allowed_modules)
            if not allowed:
                self._record_security_violation(session_id, "unauthorized_module", module_name)
                return False
        
        return True
    
    def _validate_resource_usage(self, session_id: str, resource_type: ResourceType, usage: float, **kwargs) -> bool:
        """리소스 사용량 검증"""
        if resource_type not in self.config.resource_limits:
            return True
        
        limit = self.config.resource_limits[resource_type]
        if usage > limit.max_value:
            self._record_security_violation(session_id, "resource_limit_exceeded", 
                                          f"{resource_type.value}: {usage} > {limit.max_value}")
            return False
        
        # 경고 임계값 확인
        if usage > limit.max_value * limit.warning_threshold:
            self.logger.warning(f"세션 {session_id}: {resource_type.value} 사용량 경고 "
                              f"({usage}/{limit.max_value})")
        
        return True
    
    def _validate_network_access(self, session_id: str, url: str, **kwargs) -> bool:
        """네트워크 접근 검증"""
        # 기본적으로 허용 (필요시 제한 로직 추가)
        return True
    
    def _validate_file_access(self, session_id: str, file_path: str, **kwargs) -> bool:
        """파일 접근 검증"""
        # 기본적으로 허용 (필요시 제한 로직 추가)
        return True
    
    def _record_security_violation(self, session_id: str, violation_type: str, details: str):
        """보안 위반 기록"""
        with self.session_lock:
            if session_id in self.active_sessions:
                violation = {
                    "type": violation_type,
                    "details": details,
                    "timestamp": time.time()
                }
                self.active_sessions[session_id]["security_violations"].append(violation)
                self.logger.warning(f"보안 위반 기록: {session_id} - {violation_type}: {details}")
    
    async def _start_resource_monitoring(self):
        """리소스 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.logger.info("리소스 모니터링 시작됨")
        
        # 실제 구현에서는 별도 백그라운드 태스크로 리소스 모니터링
        # 현재는 플레이스홀더로만 구현
    
    async def _stop_resource_monitoring(self):
        """리소스 모니터링 중단"""
        self.monitoring_active = False
        self.logger.info("리소스 모니터링 중단됨")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        with self.session_lock:
            return self.active_sessions.get(session_id)
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """모든 활성 세션 정보 조회"""
        with self.session_lock:
            return self.active_sessions.copy()
    
    def update_session_activity(self, session_id: str):
        """세션 활동 시간 업데이트"""
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["last_activity"] = time.time()
    
    async def cleanup_inactive_sessions(self, max_idle_time: int = 3600):
        """비활성 세션 정리 (기본: 1시간)"""
        current_time = time.time()
        inactive_sessions = []
        
        with self.session_lock:
            for session_id, session_info in self.active_sessions.items():
                if current_time - session_info["last_activity"] > max_idle_time:
                    inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            await self.destroy_isolation_session(session_id)
            self.logger.info(f"비활성 세션 정리됨: {session_id}")
    
    def __del__(self):
        """소멸자"""
        try:
            # 모든 활성 세션 정리
            for session_id in list(self.active_sessions.keys()):
                asyncio.create_task(self.destroy_isolation_session(session_id))
        except:
            pass

# 전역 인스턴스 (선택적 사용)
_global_isolation_manager = None

def get_global_isolation_manager() -> UniversalIsolationManager:
    """전역 격리 관리자 인스턴스 반환"""
    global _global_isolation_manager
    if _global_isolation_manager is None:
        _global_isolation_manager = UniversalIsolationManager()
    return _global_isolation_manager

def set_global_isolation_config(config: IsolationConfig):
    """전역 격리 설정 업데이트"""
    global _global_isolation_manager
    _global_isolation_manager = UniversalIsolationManager(config)

# 편의 함수들
async def create_isolated_session(session_id: str, agent_type: str = "generic") -> bool:
    """격리 세션 생성 (편의 함수)"""
    manager = get_global_isolation_manager()
    return await manager.create_isolation_session(session_id, agent_type)

async def destroy_isolated_session(session_id: str) -> bool:
    """격리 세션 제거 (편의 함수)"""
    manager = get_global_isolation_manager()
    return await manager.destroy_isolation_session(session_id)

def validate_isolation_policy(session_id: str, policy_type: str, **kwargs) -> bool:
    """격리 정책 검증 (편의 함수)"""
    manager = get_global_isolation_manager()
    return manager.validate_security_policy(session_id, policy_type, **kwargs)