import re
import time
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from config_manager import ConfigManager, ContaminationPattern

class ContaminationDetector(ABC):
    @abstractmethod
    def detect(self, data: Any, context: str = "") -> bool:
        pass

class TextContaminationDetector(ContaminationDetector):
    def __init__(self, patterns: List[str], case_sensitive: bool = False):
        self.patterns = patterns
        self.case_sensitive = case_sensitive
    
    def detect(self, data: Any, context: str = "") -> bool:
        if not isinstance(data, str) or not data:
            return False
        
        text = data if self.case_sensitive else data.lower()
        
        for pattern in self.patterns:
            pattern_text = pattern if self.case_sensitive else pattern.lower()
            if pattern_text in text:
                return True
        return False

class RegexContaminationDetector(ContaminationDetector):
    def __init__(self, regex_patterns: List[str], flags: int = 0):
        self.compiled_patterns = [re.compile(p, flags) for p in regex_patterns]
    
    def detect(self, data: Any, context: str = "") -> bool:
        if not isinstance(data, str):
            return False
        
        for pattern in self.compiled_patterns:
            if pattern.search(data):
                return True
        return False

class UniversalIsolationManager:
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.detectors = self._initialize_detectors()
        self.contamination_log = []
        self.custom_validators = {}
    
    def _initialize_detectors(self) -> List[ContaminationDetector]:
        detectors = []
        config = self.config_manager.config
        
        if config.contamination_keywords:
            detectors.append(TextContaminationDetector(config.contamination_keywords))
        
        if config.ai_generated_patterns:
            detectors.append(TextContaminationDetector(config.ai_generated_patterns))
        
        for custom_pattern in config.custom_patterns:
            if custom_pattern.severity == "regex":
                detectors.append(RegexContaminationDetector(custom_pattern.patterns))
            else:
                detectors.append(TextContaminationDetector(custom_pattern.patterns))
        
        return detectors
    
    def register_custom_validator(self, name: str, validator: Callable[[Any, str], bool]):
        self.custom_validators[name] = validator
    
    def is_contaminated(self, data: Any, context: str = "") -> bool:
        for detector in self.detectors:
            if detector.detect(data, context):
                self._log_contamination(detector.__class__.__name__, str(data)[:100], context)
                return True
        
        for name, validator in self.custom_validators.items():
            if validator(data, context):
                self._log_contamination(f"custom_{name}", str(data)[:100], context)
                return True
        
        if isinstance(data, dict):
            return self._check_dict_contamination(data, context)
        elif isinstance(data, list):
            return any(self.is_contaminated(item, f"{context}[{i}]") for i, item in enumerate(data))
        
        return False
    
    def _check_dict_contamination(self, data: dict, context: str = "") -> bool:
        contamination_indicators = ["fallback_used", "ai_generated", "synthetic_data"]
        
        for indicator in contamination_indicators:
            if data.get(indicator) or data.get("metadata", {}).get(indicator):
                self._log_contamination("metadata_indicator", indicator, context)
                return True
        
        text_fields = self._get_text_fields(data)
        for field in text_fields:
            if field in data and self.is_contaminated(data[field], f"{context}.{field}"):
                return True
        
        return False
    
    def _get_text_fields(self, data: dict) -> List[str]:
        common_text_fields = ["title", "subtitle", "body", "content", "description", "text", "message"]
        detected_fields = []
        
        for key in data.keys():
            if isinstance(data[key], str) and (key.lower() in common_text_fields or 
                                             any(field in key.lower() for field in ["text", "content", "description"])):
                detected_fields.append(key)
        
        return detected_fields
    
    def _log_contamination(self, contamination_type: str, detected_content: str, context: str):
        if self.config_manager.config.enable_logging:
            log_entry = {
                "type": contamination_type,
                "content": detected_content[:100],
                "context": context,
                "timestamp": time.time(),
                "language": self.config_manager.config.language,
                "domain": self.config_manager.config.domain
            }
            self.contamination_log.append(log_entry)
    
    def filter_contaminated_data(self, data_list: List[Any], context: str = "") -> List[Any]:
        if not isinstance(data_list, list):
            return data_list
        
        clean_data = []
        contaminated_count = 0
        
        for i, item in enumerate(data_list):
            if not self.is_contaminated(item, f"{context}[{i}]"):
                clean_data.append(item)
            else:
                contaminated_count += 1
        
        if contaminated_count > 0 and self.config_manager.config.enable_logging:
            print(f"🛡️ Isolation Applied: {contaminated_count} contaminated items removed, {len(clean_data)} clean items preserved")
        
        return clean_data
    
    def validate_preservation_rate(self, result: Any, original: str, context: str = "") -> Dict[str, Any]:
        if not isinstance(result, (dict, str)) or not original:
            return {"preservation_rate": 0.0, "contamination_detected": True}
        
        original_tokens = self._tokenize_text(original)
        result_text = self._extract_text_from_result(result)
        result_tokens = self._tokenize_text(result_text)
        
        if not original_tokens:
            return {"preservation_rate": 0.0, "contamination_detected": True}
        
        preserved_tokens = original_tokens.intersection(result_tokens)
        preservation_rate = len(preserved_tokens) / len(original_tokens)
        contamination_detected = self.is_contaminated(result, context)
        
        return {
            "preservation_rate": preservation_rate,
            "original_tokens": len(original_tokens),
            "preserved_tokens": len(preserved_tokens),
            "contamination_detected": contamination_detected,
            "meets_threshold": preservation_rate >= self.config_manager.config.preservation_threshold,
            "context": context
        }
    
    def _tokenize_text(self, text: str) -> set:
        if not text:
            return set()
        
        tokens = re.findall(r'\w+', text.lower())
        return set(tokens)
    
    def _extract_text_from_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            text_parts = []
            text_fields = self._get_text_fields(result)
            for field in text_fields:
                if field in result:
                    text_parts.append(str(result[field]))
            return " ".join(text_parts)
        else:
            return str(result)
    
    def get_contamination_report(self) -> Dict[str, Any]:
        if not self.contamination_log:
            return {"total_contaminations": 0, "types": {}, "recent_detections": []}
        
        types_count = {}
        for entry in self.contamination_log:
            contamination_type = entry["type"]
            types_count[contamination_type] = types_count.get(contamination_type, 0) + 1
        
        return {
            "total_contaminations": len(self.contamination_log),
            "types": types_count,
            "recent_detections": self.contamination_log[-10:],
            "config": {
                "language": self.config_manager.config.language,
                "domain": self.config_manager.config.domain,
                "preservation_threshold": self.config_manager.config.preservation_threshold
            }
        }
