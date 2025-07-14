import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from isolation_manager import UniversalIsolationManager
from agent_isolation_plugins import PluginManager
from session_manager import UniversalSessionManager

@dataclass
class DataTransferRequest:
    source_agent: str
    target_agent: str
    data: Any
    transfer_type: str
    session_id: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class UniversalCommunicationIsolator:
    def __init__(self, isolation_manager: UniversalIsolationManager = None,
                 session_manager: UniversalSessionManager = None):
        self.isolation_manager = isolation_manager or UniversalIsolationManager()
        self.session_manager = session_manager or UniversalSessionManager()
        self.plugin_manager = PluginManager(self.isolation_manager)
        self.transfer_log = []
        self.blocked_transfers = []
        self.transfer_rules = {}
    
    def add_transfer_rule(self, source_pattern: str, target_pattern: str, 
                         allowed: bool = True, custom_validator: callable = None):
        rule_key = f"{source_pattern}->{target_pattern}"
        self.transfer_rules[rule_key] = {
            "allowed": allowed,
            "validator": custom_validator
        }
    
    def transfer_data(self, request: DataTransferRequest) -> Dict[str, Any]:
        if not self._validate_transfer_rules(request):
            return self._create_blocked_response(request, "transfer_rule_violation")
        
        if self.isolation_manager.is_contaminated(request.data, f"{request.source_agent}_to_{request.target_agent}"):
            return self._create_blocked_response(request, "contamination_detected")
        
        if not self._validate_session_isolation(request):
            return self._create_blocked_response(request, "session_isolation_violation")
        
        cleaned_data = self._clean_transfer_data(request)
        
        self._log_successful_transfer(request, cleaned_data)
        
        return {
            "success": True,
            "cleaned_data": cleaned_data,
            "isolation_metadata": {
                "contamination_filtered": True,
                "session_isolated": True,
                "transfer_id": len(self.transfer_log),
                "plugins_applied": self._get_applied_plugins(request)
            }
        }
    
    def _validate_transfer_rules(self, request: DataTransferRequest) -> bool:
        for rule_pattern, rule_config in self.transfer_rules.items():
            source_pattern, target_pattern = rule_pattern.split('->')
            
            if (source_pattern in request.source_agent and 
                target_pattern in request.target_agent):
                
                if not rule_config["allowed"]:
                    return False
                
                if rule_config["validator"]:
                    return rule_config["validator"](request)
        
        return True
    
    def _validate_session_isolation(self, request: DataTransferRequest) -> bool:
        return self.session_manager.validate_cross_agent_communication(
            request.source_agent, request.target_agent, request.session_id
        )
    
    def _clean_transfer_data(self, request: DataTransferRequest) -> Any:
        source_plugin = self.plugin_manager.get_plugin_for_agent(
            self._extract_agent_type(request.source_agent)
        )
        target_plugin = self.plugin_manager.get_plugin_for_agent(
            self._extract_agent_type(request.target_agent)
        )
        
        cleaned_data = request.data
        
        if source_plugin:
            cleaned_data = source_plugin.isolate_agent_data(
                cleaned_data, f"source_{request.source_agent}"
            )
        
        if target_plugin:
            cleaned_data = target_plugin.isolate_agent_data(
                cleaned_data, f"target_{request.target_agent}"
            )
        
        if isinstance(cleaned_data, dict):
            cleaned_data["_transfer_metadata"] = {
                "source_agent": request.source_agent,
                "target_agent": request.target_agent,
                "transfer_timestamp": request.timestamp,
                "session_id": request.session_id,
                "isolation_applied": True
            }
        
        return cleaned_data
    
    def _extract_agent_type(self, agent_name: str) -> str:
        common_suffixes = ["Agent", "Bot", "Service", "Handler", "Processor"]
        
        for suffix in common_suffixes:
            if agent_name.endswith(suffix):
                return agent_name
        
        return f"{agent_name}Agent"
    
    def _create_blocked_response(self, request: DataTransferRequest, reason: str) -> Dict[str, Any]:
        self.blocked_transfers.append({
            "request": request,
            "reason": reason,
            "timestamp": time.time()
        })
        
        return {
            "success": False,
            "reason": reason,
            "fallback_data": self._generate_fallback_data(request)
        }
    
    def _generate_fallback_data(self, request: DataTransferRequest) -> Dict[str, Any]:
        return {
            "source_agent": request.source_agent,
            "target_agent": request.target_agent,
            "fallback_reason": "communication_blocked",
            "session_id": request.session_id,
            "timestamp": request.timestamp,
            "isolation_applied": True,
            "available_data": {}
        }
    
    def _log_successful_transfer(self, request: DataTransferRequest, cleaned_data: Any):
        self.transfer_log.append({
            "source_agent": request.source_agent,
            "target_agent": request.target_agent,
            "transfer_type": request.transfer_type,
            "session_id": request.session_id,
            "data_size": len(str(cleaned_data)),
            "timestamp": request.timestamp,
            "isolation_applied": True
        })
    
    def _get_applied_plugins(self, request: DataTransferRequest) -> List[str]:
        applied_plugins = []
        
        source_plugin = self.plugin_manager.get_plugin_for_agent(
            self._extract_agent_type(request.source_agent)
        )
        if source_plugin:
            applied_plugins.append(source_plugin.plugin_name)
        
        target_plugin = self.plugin_manager.get_plugin_for_agent(
            self._extract_agent_type(request.target_agent)
        )
        if target_plugin and target_plugin.plugin_name not in applied_plugins:
            applied_plugins.append(target_plugin.plugin_name)
        
        return applied_plugins
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        total_transfers = len(self.transfer_log)
        blocked_transfers = len(self.blocked_transfers)
        total_attempts = total_transfers + blocked_transfers
        
        return {
            "total_transfers": total_transfers,
            "successful_transfers": total_transfers,
            "blocked_transfers": blocked_transfers,
            "success_rate": (total_transfers / total_attempts * 100) if total_attempts > 0 else 0,
            "isolation_effectiveness": (blocked_transfers / total_attempts * 100) if total_attempts > 0 else 0,
            "active_plugins": len(self.plugin_manager.plugins),
            "transfer_rules": len(self.transfer_rules)
        }

class UniversalAgentCommunicationMixin:
    def __init_universal_communication__(self, agent_type: str = None):
        self.communication_isolator = UniversalCommunicationIsolator()
        self.agent_name = self.__class__.__name__
        self.agent_type = agent_type or self._infer_agent_type()
    
    def _infer_agent_type(self) -> str:
        class_name = self.__class__.__name__
        if "Text" in class_name or "Content" in class_name:
            return "TextAgent"
        elif "Image" in class_name or "Media" in class_name:
            return "ImageAgent"
        elif "Data" in class_name or "Process" in class_name:
            return "DataAgent"
        else:
            return "GenericAgent"
    
    def send_data_to_agent(self, target_agent: str, data: Any,
                          transfer_type: str = "result", session_id: str = None,
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if session_id is None:
            session_id = getattr(self, 'current_session_id', 'default_session')
        
        request = DataTransferRequest(
            source_agent=self.agent_name,
            target_agent=target_agent,
            data=data,
            transfer_type=transfer_type,
            session_id=session_id,
            timestamp=time.time(),
            metadata=metadata
        )
        
        return self.communication_isolator.transfer_data(request)
    
    def receive_data_from_agent(self, source_agent: str, data: Any) -> Any:
        if self.communication_isolator.isolation_manager.is_contaminated(
            data, f"{source_agent}_to_{self.agent_name}_receive"
        ):
            return None
        
        return data
    
    def get_communication_stats(self) -> Dict[str, Any]:
        return self.communication_isolator.get_transfer_statistics()
