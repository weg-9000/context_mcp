import os
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from isolation_manager import UniversalIsolationManager

@dataclass
class SessionConfig:
    session_id: str
    isolation_level: str
    data_retention_hours: int
    enable_cross_session_learning: bool
    custom_validators: Dict[str, Callable]
    domain: str
    language: str

class UniversalSessionManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, isolation_manager: UniversalIsolationManager = None):
        if not hasattr(self, 'initialized'):
            self.sessions = {}
            self.session_locks = {}
            self.session_data = {}
            self.isolation_manager = isolation_manager
            self.session_configs = {}
            self.initialized = True
    
    def create_session(self, session_id: str = None, config: SessionConfig = None) -> str:
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000000)}"
        
        if session_id not in self.sessions:
            self.sessions[session_id] = True
            self.session_locks[session_id] = threading.Lock()
            self.session_data[session_id] = {
                "created_at": time.time(),
                "agent_results": {},
                "contamination_log": [],
                "communication_log": []
            }
            
            self.session_configs[session_id] = config or self._get_default_session_config(session_id)
        
        return session_id
    
    def _get_default_session_config(self, session_id: str) -> SessionConfig:
        return SessionConfig(
            session_id=session_id,
            isolation_level="moderate",
            data_retention_hours=24,
            enable_cross_session_learning=False,
            custom_validators={},
            domain="general",
            language="en"
        )
    
    def validate_cross_agent_communication(self, source_agent: str, target_agent: str, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        
        config = self.session_configs[session_id]
        
        if config.isolation_level == "strict":
            return self._validate_strict_communication(source_agent, target_agent, session_id)
        elif config.isolation_level == "moderate":
            return self._validate_moderate_communication(source_agent, target_agent, session_id)
        else:
            return True
    
    def _validate_strict_communication(self, source_agent: str, target_agent: str, session_id: str) -> bool:
        source_session = self._get_agent_session(source_agent)
        target_session = self._get_agent_session(target_agent)
        return source_session == target_session == session_id
    
    def _validate_moderate_communication(self, source_agent: str, target_agent: str, session_id: str) -> bool:
        config = self.session_configs[session_id]
        
        for validator_name, validator in config.custom_validators.items():
            if not validator(source_agent, target_agent, session_id):
                return False
        
        return True
    
    def _get_agent_session(self, agent_name: str) -> str:
        return getattr(threading.current_thread(), 'session_id', 'default_session')
    
    def store_agent_result(self, session_id: str, agent_name: str, result: Any) -> bool:
        if session_id not in self.sessions:
            return False
        
        with self.session_locks[session_id]:
            if self.isolation_manager and self.isolation_manager.is_contaminated(result, f"{agent_name}_result"):
                self.session_data[session_id]["contamination_log"].append({
                    "agent": agent_name,
                    "timestamp": time.time(),
                    "contamination_type": "agent_result"
                })
                return False
            
            if agent_name not in self.session_data[session_id]["agent_results"]:
                self.session_data[session_id]["agent_results"][agent_name] = []
            
            self.session_data[session_id]["agent_results"][agent_name].append({
                "timestamp": time.time(),
                "result": result,
                "isolation_verified": True
            })
            
            return True
    
    def get_agent_results(self, session_id: str, agent_name: str, max_results: int = None) -> List[Any]:
        if session_id not in self.sessions:
            return []
        
        with self.session_locks[session_id]:
            results = self.session_data[session_id]["agent_results"].get(agent_name, [])
            config = self.session_configs[session_id]
            
            filtered_results = self._filter_results_by_config(results, config)
            
            if max_results:
                filtered_results = filtered_results[:max_results]
            
            return [r["result"] for r in filtered_results]
    
    def _filter_results_by_config(self, results: List[Dict], config: SessionConfig) -> List[Dict]:
        if config.isolation_level == "strict":
            return results
        elif config.isolation_level == "moderate":
            current_time = time.time()
            return [r for r in results if current_time - r["timestamp"] < 3600]
        else:
            return results
    
    def cleanup_expired_sessions(self):
        current_time = time.time()
        expired_sessions = []
        
        for session_id, config in self.session_configs.items():
            session_age = current_time - self.session_data[session_id]["created_at"]
            if session_age > config.data_retention_hours * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
    
    def _cleanup_session(self, session_id: str):
        with self._lock:
            for storage in [self.sessions, self.session_data, self.session_locks, self.session_configs]:
                if session_id in storage:
                    del storage[session_id]

class UniversalSessionAwareMixin:
    def __init_universal_session_awareness__(self, session_id: Optional[str] = None, 
                                           session_config: SessionConfig = None):
        self.session_manager = UniversalSessionManager()
        self.current_session_id = session_id or self.session_manager.create_session(config=session_config)
        self.agent_name = self.__class__.__name__
    
    def store_result(self, result: Any) -> bool:
        return self.session_manager.store_agent_result(
            self.current_session_id, self.agent_name, result
        )
    
    def get_previous_results(self, max_results: int = 10) -> List[Any]:
        return self.session_manager.get_agent_results(
            self.current_session_id, self.agent_name, max_results
        )
    
    def get_session_isolated_identifier(self, base_identifier: str) -> str:
        return f"{base_identifier}_{self.current_session_id}"
