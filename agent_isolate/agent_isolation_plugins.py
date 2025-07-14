from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from isolation_manager import UniversalIsolationManager

class AgentIsolationPlugin(ABC):
    def __init__(self, isolation_manager: UniversalIsolationManager):
        self.isolation_manager = isolation_manager
        self.plugin_name = self.__class__.__name__
    
    @abstractmethod
    def isolate_agent_data(self, data: Any, context: str = "") -> Any:
        pass
    
    @abstractmethod
    def get_supported_agent_types(self) -> List[str]:
        pass

class TextProcessingAgentPlugin(AgentIsolationPlugin):
    def isolate_agent_data(self, data: Any, context: str = "") -> Any:
        if isinstance(data, dict):
            return self._isolate_text_dict(data, context)
        elif isinstance(data, list):
            return self.isolation_manager.filter_contaminated_data(data, context)
        elif isinstance(data, str):
            return data if not self.isolation_manager.is_contaminated(data, context) else ""
        return data
    
    def _isolate_text_dict(self, data: dict, context: str) -> dict:
        clean_data = {}
        for key, value in data.items():
            if not self.isolation_manager.is_contaminated(value, f"{context}.{key}"):
                clean_data[key] = value
        
        clean_data["_isolation_metadata"] = {
            "plugin": self.plugin_name,
            "context": context,
            "isolation_applied": True
        }
        return clean_data
    
    def get_supported_agent_types(self) -> List[str]:
        return ["TextAgent", "ContentAgent", "WritingAgent", "AnalysisAgent"]

class ImageProcessingAgentPlugin(AgentIsolationPlugin):
    def isolate_agent_data(self, data: Any, context: str = "") -> Any:
        if isinstance(data, dict):
            return self._isolate_image_dict(data, context)
        elif isinstance(data, list):
            return self._isolate_image_list(data, context)
        return data
    
    def _isolate_image_dict(self, data: dict, context: str) -> dict:
        clean_data = {}
        
        for key, value in data.items():
            if key.lower() in ['image_url', 'image_urls', 'images']:
                clean_data[key] = self._filter_trusted_urls(value)
            elif not self.isolation_manager.is_contaminated(value, f"{context}.{key}"):
                clean_data[key] = value
        
        return clean_data
    
    def _isolate_image_list(self, data: list, context: str) -> list:
        clean_images = []
        max_items = self.isolation_manager.config_manager.config.max_items_per_section
        
        for i, item in enumerate(data[:max_items]):
            if not self.isolation_manager.is_contaminated(item, f"{context}[{i}]"):
                clean_images.append(item)
        
        return clean_images
    
    def _filter_trusted_urls(self, urls: Any) -> Any:
        trusted_domains = self.isolation_manager.config_manager.config.trusted_domains
        
        if isinstance(urls, str):
            return urls if any(domain in urls for domain in trusted_domains) else ""
        elif isinstance(urls, list):
            return [url for url in urls if any(domain in url for domain in trusted_domains)]
        
        return urls
    
    def get_supported_agent_types(self) -> List[str]:
        return ["ImageAgent", "LayoutAgent", "DesignAgent", "MediaAgent"]

class DataProcessingAgentPlugin(AgentIsolationPlugin):
    def isolate_agent_data(self, data: Any, context: str = "") -> Any:
        if isinstance(data, dict):
            return self._isolate_data_dict(data, context)
        elif isinstance(data, list):
            return self._isolate_data_list(data, context)
        return data
    
    def _isolate_data_dict(self, data: dict, context: str) -> dict:
        clean_data = {}
        
        for key, value in data.items():
            if key.startswith('_') or key.lower() in ['metadata', 'config', 'settings']:
                if not self.isolation_manager.is_contaminated(value, f"{context}.{key}"):
                    clean_data[key] = value
            else:
                clean_data[key] = self.isolate_agent_data(value, f"{context}.{key}")
        
        return clean_data
    
    def _isolate_data_list(self, data: list, context: str) -> list:
        return [
            self.isolate_agent_data(item, f"{context}[{i}]")
            for i, item in enumerate(data)
            if not self.isolation_manager.is_contaminated(item, f"{context}[{i}]")
        ]
    
    def get_supported_agent_types(self) -> List[str]:
        return ["DataAgent", "ProcessingAgent", "TransformAgent", "FilterAgent"]

class PluginManager:
    def __init__(self, isolation_manager: UniversalIsolationManager):
        self.isolation_manager = isolation_manager
        self.plugins = {}
        self._register_default_plugins()
    
    def _register_default_plugins(self):
        default_plugins = [
            TextProcessingAgentPlugin(self.isolation_manager),
            ImageProcessingAgentPlugin(self.isolation_manager),
            DataProcessingAgentPlugin(self.isolation_manager)
        ]
        
        for plugin in default_plugins:
            self.register_plugin(plugin)
    
    def register_plugin(self, plugin: AgentIsolationPlugin):
        for agent_type in plugin.get_supported_agent_types():
            self.plugins[agent_type] = plugin
    
    def get_plugin_for_agent(self, agent_type: str) -> Optional[AgentIsolationPlugin]:
        return self.plugins.get(agent_type)
    
    def isolate_agent_data(self, agent_type: str, data: Any, context: str = "") -> Any:
        plugin = self.get_plugin_for_agent(agent_type)
        if plugin:
            return plugin.isolate_agent_data(data, context)
        else:
            return self.isolation_manager.filter_contaminated_data(data, context) if isinstance(data, list) else data
