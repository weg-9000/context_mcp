import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

@dataclass
class ContaminationPattern:
    name: str
    patterns: List[str]
    language: str
    domain: str
    severity: str

@dataclass
class IsolationConfig:
    contamination_keywords: List[str]
    ai_generated_patterns: List[str]
    trusted_domains: List[str]
    preservation_threshold: float
    max_items_per_section: int
    enable_logging: bool
    language: str
    domain: str
    custom_patterns: List[ContaminationPattern]

class ConfigManager:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "./isolation_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> IsolationConfig:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                return self._dict_to_config(config_data)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> IsolationConfig:
        return IsolationConfig(
            contamination_keywords=[],
            ai_generated_patterns=[
                "generated content pattern",
                "artificial intelligence created",
                "automatically generated"
            ],
            trusted_domains=["trusted-domain.com"],
            preservation_threshold=0.3,
            max_items_per_section=5,
            enable_logging=True,
            language="en",
            domain="general",
            custom_patterns=[]
        )
    
    def _dict_to_config(self, config_dict: Dict) -> IsolationConfig:
        custom_patterns = []
        for pattern_data in config_dict.get('custom_patterns', []):
            custom_patterns.append(ContaminationPattern(**pattern_data))
        
        config_dict['custom_patterns'] = custom_patterns
        return IsolationConfig(**config_dict)
    
    def save_config(self):
        config_dict = asdict(self.config)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
    
    def add_contamination_pattern(self, pattern: ContaminationPattern):
        self.config.custom_patterns.append(pattern)
        self.save_config()

class LanguagePatternLoader:
    @staticmethod
    def load_patterns(language: str, domain: str = "general") -> List[ContaminationPattern]:
        patterns_file = f"./patterns/{language}_{domain}.json"
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
                return [ContaminationPattern(**p) for p in patterns_data]
        return []
