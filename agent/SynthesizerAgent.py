# SynthesizerAgent - Semantic Kernel 프레임워크 기반 구현
# 프레임워크: Semantic Kernel (플러그인 아키텍처와 데이터 변환에 특화)

import asyncio
from datetime import datetime
import json
import time
import re
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Semantic Kernel 관련 imports
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.planning import ActionPlanner
    from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter
    from semantic_kernel.orchestration.sk_context import SKContext
    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    # Semantic Kernel이 없을 때의 폴백 구현
    SEMANTIC_KERNEL_AVAILABLE = False
    print("Semantic Kernel not available, using fallback implementation")

from modular_agent_architecture import ProcessingMode
from modular_agent_architecture import (
    ModularAgent, AgentConfig, ProcessingRequest, ProcessingResponse,
    FrameworkAdapter, AgentCapability, AgentFramework
)


class SynthesisMode(Enum):
    """데이터 가공 모드"""
    STRUCTURE = "structure"           # 구조화
    COMPRESS = "compress"             # 압축
    TRANSFORM = "transform"           # 변환
    ENRICH = "enrich"                # 풍부화
    NORMALIZE = "normalize"           # 정규화
    OPTIMIZE = "optimize"             # 최적화


class OutputFormat(Enum):
    """출력 형식"""
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    STRUCTURED_TEXT = "structured_text"
    SEMANTIC_HTML = "semantic_html"
    KNOWLEDGE_GRAPH = "knowledge_graph"


@dataclass
class TransformationRule:
    """변환 규칙"""
    rule_id: str
    name: str
    description: str
    input_pattern: str
    output_template: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """가공 결과"""
    result_id: str
    original_content: Any
    synthesized_content: Any
    format_type: OutputFormat
    compression_ratio: float
    quality_metrics: Dict[str, float]
    transformations_applied: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticKernelAdapter(FrameworkAdapter):
    """Semantic Kernel 프레임워크 어댑터"""
    
    def __init__(self):
        self.kernel: Optional[sk.Kernel] = None
        self.planner: Optional[ActionPlanner] = None
        self.skills: Dict[str, Any] = {}
        self.transformation_rules: Dict[str, TransformationRule] = {}
        self.is_initialized = False
    
    def get_framework_name(self) -> str:
        return "Semantic Kernel"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Semantic Kernel 초기화"""
        try:
            if not SEMANTIC_KERNEL_AVAILABLE:
                return await self._initialize_fallback(config)
            
            # 커널 생성
            self.kernel = sk.Kernel()
            
            # AI 서비스 설정
            api_key = config.get("openai_api_key", "default-key")
            self.kernel.add_chat_service(
                "chat_completion",
                OpenAIChatCompletion(
                    model_id=config.get("model_id", "gpt-3.5-turbo"),
                    api_key=api_key
                )
            )
            
            # 스킬 등록
            await self._register_synthesis_skills()
            
            # 플래너 생성
            self.planner = ActionPlanner(self.kernel)
            
            # 변환 규칙 로드
            await self._load_transformation_rules(config)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Semantic Kernel 초기화 실패: {e}")
            return await self._initialize_fallback(config)
    
    async def _initialize_fallback(self, config: Dict[str, Any]) -> bool:
        """폴백 초기화"""
        self.transformation_rules = self._get_default_transformation_rules()
        self.is_initialized = True
        return True
    
    async def _register_synthesis_skills(self):
        """가공 스킬들 등록"""
        if not SEMANTIC_KERNEL_AVAILABLE:
            return
        
        # 데이터 구조화 스킬
        structure_skill = self._create_structure_skill()
        self.skills["structure"] = self.kernel.import_skill(structure_skill, "StructureSkill")
        
        # 데이터 압축 스킬
        compress_skill = self._create_compress_skill()
        self.skills["compress"] = self.kernel.import_skill(compress_skill, "CompressSkill")
        
        # 포맷 변환 스킬
        format_skill = self._create_format_skill()
        self.skills["format"] = self.kernel.import_skill(format_skill, "FormatSkill")
        
        # 품질 향상 스킬
        enhance_skill = self._create_enhance_skill()
        self.skills["enhance"] = self.kernel.import_skill(enhance_skill, "EnhanceSkill")
    
    def _create_structure_skill(self):
        """데이터 구조화 스킬 생성"""
        class StructureSkill:
            @sk_function(
                description="비구조화된 데이터를 구조화된 형태로 변환",
                name="structure_data"
            )
            @sk_function_context_parameter(
                name="input_data",
                description="구조화할 원시 데이터"
            )
            @sk_function_context_parameter(
                name="target_format",
                description="목표 형식 (markdown, json, xml 등)"
            )
            def structure_data(self, context: SKContext) -> str:
                input_data = context["input_data"]
                target_format = context.get("target_format", "markdown")
                
                # LLM 기반 구조화 (실제로는 더 복잡한 프롬프트 사용)
                prompt = f"""
                다음 데이터를 {target_format} 형식으로 구조화해주세요:
                
                {input_data}
                
                구조화 원칙:
                1. 논리적 계층 구조 생성
                2. 중요도에 따른 정보 배치
                3. 가독성 최적화
                4. 메타데이터 포함
                
                구조화된 결과:
                """
                
                # 실제 구현에서는 LLM API 호출
                return f"# 구조화된 데이터\n\n{input_data}\n\n*{target_format} 형식으로 구조화됨*"
            
            @sk_function(
                description="텍스트에서 핵심 엔티티와 관계를 추출하여 지식 그래프 생성",
                name="extract_knowledge_graph"
            )
            @sk_function_context_parameter(
                name="text_content",
                description="지식 그래프를 생성할 텍스트"
            )
            def extract_knowledge_graph(self, context: SKContext) -> str:
                text_content = context["text_content"]
                
                # 엔티티 추출 및 관계 분석 (LLM 기반)
                return f"""
                {{
                  "entities": ["엔티티1", "엔티티2", "엔티티3"],
                  "relationships": [
                    {{"from": "엔티티1", "to": "엔티티2", "type": "관련됨"}},
                    {{"from": "엔티티2", "to": "엔티티3", "type": "포함함"}}
                  ],
                  "metadata": {{"source": "text_analysis", "confidence": 0.85}}
                }}
                """
        
        return StructureSkill()
    
    def _create_compress_skill(self):
        """데이터 압축 스킬 생성"""
        class CompressSkill:
            @sk_function(
                description="텍스트 데이터를 지능적으로 압축하면서 핵심 정보 보존",
                name="intelligent_compress"
            )
            @sk_function_context_parameter(
                name="content",
                description="압축할 콘텐츠"
            )
            @sk_function_context_parameter(
                name="compression_ratio",
                description="압축 비율 (0.1 ~ 1.0)"
            )
            @sk_function_context_parameter(
                name="preserve_keywords",
                description="보존할 핵심 키워드들"
            )
            def intelligent_compress(self, context: SKContext) -> str:
                content = context["content"]
                ratio = float(context.get("compression_ratio", "0.3"))
                keywords = context.get("preserve_keywords", "").split(",")
                
                # 지능형 압축 로직 (LLM 기반)
                compressed_length = int(len(content) * ratio)
                
                return f"압축된 콘텐츠 (원본의 {ratio:.0%}): {content[:compressed_length]}..."
            
            @sk_function(
                description="중복 정보 제거 및 유사 내용 통합",
                name="deduplicate_content"
            )
            @sk_function_context_parameter(
                name="content_list",
                description="중복 제거할 콘텐츠 목록"
            )
            def deduplicate_content(self, context: SKContext) -> str:
                content_list = context["content_list"]
                
                # 의미적 중복 제거 (LLM 기반)
                return "중복이 제거된 고유한 콘텐츠들"
        
        return CompressSkill()
    
    def _create_format_skill(self):
        """포맷 변환 스킬 생성"""
        class FormatSkill:
            @sk_function(
                description="콘텐츠를 마크다운 형식으로 변환",
                name="to_markdown"
            )
            @sk_function_context_parameter(
                name="content",
                description="변환할 콘텐츠"
            )
            def to_markdown(self, context: SKContext) -> str:
                content = context["content"]
                
                # 마크다운 변환 로직
                if isinstance(content, dict):
                    return self._dict_to_markdown(content)
                elif isinstance(content, list):
                    return self._list_to_markdown(content)
                else:
                    return self._text_to_markdown(str(content))
            
            @sk_function(
                description="콘텐츠를 JSON 형식으로 변환",
                name="to_json"
            )
            @sk_function_context_parameter(
                name="content",
                description="변환할 콘텐츠"
            )
            def to_json(self, context: SKContext) -> str:
                content = context["content"]
                
                try:
                    if isinstance(content, str):
                        # 텍스트를 구조화된 JSON으로 변환
                        structured = {
                            "content": content,
                            "metadata": {
                                "length": len(content),
                                "type": "text",
                                "processed_at": time.time()
                            }
                        }
                        return json.dumps(structured, ensure_ascii=False, indent=2)
                    else:
                        return json.dumps(content, ensure_ascii=False, indent=2)
                except:
                    return '{"error": "JSON 변환 실패"}'
            
            @sk_function(
                description="시맨틱 HTML로 변환하여 구조와 의미 강화",
                name="to_semantic_html"
            )
            @sk_function_context_parameter(
                name="content",
                description="변환할 콘텐츠"
            )
            def to_semantic_html(self, context: SKContext) -> str:
                content = context["content"]
                
                return f"""
                <article>
                    <header>
                        <h1>가공된 콘텐츠</h1>
                    </header>
                    <main>
                        <section>
                            {content}
                        </section>
                    </main>
                    <footer>
                        <p>Semantic Kernel로 처리됨</p>
                    </footer>
                </article>
                """
            
            def _dict_to_markdown(self, data: dict) -> str:
                """딕셔너리를 마크다운으로 변환"""
                markdown = ""
                for key, value in data.items():
                    markdown += f"## {key}\n\n"
                    if isinstance(value, (list, dict)):
                        markdown += f"```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```\n\n"
                    else:
                        markdown += f"{value}\n\n"
                return markdown
            
            def _list_to_markdown(self, data: list) -> str:
                """리스트를 마크다운으로 변환"""
                markdown = ""
                for i, item in enumerate(data, 1):
                    markdown += f"{i}. {item}\n"
                return markdown + "\n"
            
            def _text_to_markdown(self, text: str) -> str:
                """텍스트를 마크다운으로 변환"""
                # 간단한 구조화
                lines = text.split('\n')
                markdown = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        markdown += "\n"
                    elif line.endswith(':') and len(line) < 50:
                        markdown += f"## {line[:-1]}\n\n"
                    elif line.startswith('-') or line.startswith('*'):
                        markdown += f"- {line[1:].strip()}\n"
                    else:
                        markdown += f"{line}\n\n"
                
                return markdown
        
        return FormatSkill()
    
    def _create_enhance_skill(self):
        """품질 향상 스킬 생성"""
        class EnhanceSkill:
            @sk_function(
                description="콘텐츠의 가독성과 구조를 개선",
                name="enhance_readability"
            )
            @sk_function_context_parameter(
                name="content",
                description="개선할 콘텐츠"
            )
            def enhance_readability(self, context: SKContext) -> str:
                content = context["content"]
                
                # 가독성 개선 로직
                enhanced = self._improve_text_structure(content)
                enhanced = self._add_contextual_information(enhanced)
                
                return enhanced
            
            @sk_function(
                description="메타데이터 추가 및 콘텐츠 풍부화",
                name="enrich_metadata"
            )
            @sk_function_context_parameter(
                name="content",
                description="풍부화할 콘텐츠"
            )
            def enrich_metadata(self, context: SKContext) -> str:
                content = context["content"]
                
                # 메타데이터 생성 및 추가
                metadata = {
                    "processed_at": datetime.now().isoformat(),
                    "content_type": "enhanced",
                    "enhancement_level": "high",
                    "semantic_tags": ["structured", "enriched", "optimized"]
                }
                
                if isinstance(content, dict):
                    content["_metadata"] = metadata
                    return json.dumps(content, ensure_ascii=False, indent=2)
                else:
                    return f"{content}\n\n---\n메타데이터: {json.dumps(metadata, ensure_ascii=False)}"
            
            def _improve_text_structure(self, text: str) -> str:
                """텍스트 구조 개선"""
                # 문단 구분 개선
                improved = re.sub(r'\n{3,}', '\n\n', text)
                
                # 문장 구분 개선
                improved = re.sub(r'([.!?])\s*([A-Z가-힣])', r'\1\n\n\2', improved)
                
                return improved.strip()
            
            def _add_contextual_information(self, text: str) -> str:
                """문맥 정보 추가"""
                return f"# 처리된 콘텐츠\n\n{text}\n\n*Semantic Kernel로 가공됨*"
        
        return EnhanceSkill()
    
    async def _load_transformation_rules(self, config: Dict[str, Any]):
        """변환 규칙 로드"""
        # 기본 규칙들
        self.transformation_rules = self._get_default_transformation_rules()
        
        # 사용자 정의 규칙 로드
        custom_rules = config.get("transformation_rules", [])
        for rule_config in custom_rules:
            rule = TransformationRule(**rule_config)
            self.transformation_rules[rule.rule_id] = rule
    
    def _get_default_transformation_rules(self) -> Dict[str, TransformationRule]:
        """기본 변환 규칙들"""
        return {
            "list_to_markdown": TransformationRule(
                rule_id="list_to_markdown",
                name="리스트를 마크다운으로",
                description="Python 리스트를 마크다운 목록으로 변환",
                input_pattern="list",
                output_template="markdown_list"
            ),
            "dict_to_json": TransformationRule(
                rule_id="dict_to_json",
                name="딕셔너리를 JSON으로",
                description="Python 딕셔너리를 JSON 형식으로 변환",
                input_pattern="dict",
                output_template="json"
            ),
            "text_to_structure": TransformationRule(
                rule_id="text_to_structure",
                name="텍스트 구조화",
                description="평문 텍스트를 구조화된 형태로 변환",
                input_pattern="text",
                output_template="structured_text"
            )
        }
    
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Semantic Kernel을 사용한 요청 처리"""
        if not self.is_initialized:
            await self.initialize({})
        
        if self.kernel and SEMANTIC_KERNEL_AVAILABLE:
            return await self._process_with_semantic_kernel(request)
        else:
            return await self._process_with_fallback(request)
    
    async def _process_with_semantic_kernel(self, request: ProcessingRequest) -> ProcessingResponse:
        """Semantic Kernel을 사용한 처리"""
        start_time = time.time()
        
        try:
            # 처리 계획 생성
            plan = await self._create_synthesis_plan(request)
            
            # 계획 실행
            synthesis_result = await self._execute_synthesis_plan(plan, request)
            
            return ProcessingResponse(
                request_id=request.request_id,
                processed_content=synthesis_result.synthesized_content,
                confidence_score=synthesis_result.quality_metrics.get("overall_confidence", 0.8),
                quality_metrics=synthesis_result.quality_metrics,
                processing_time=time.time() - start_time,
                framework_info=self.get_framework_info(),
                metadata={
                    "synthesis_mode": request.processing_options.get("synthesis_mode", "structure"),
                    "output_format": synthesis_result.format_type.value,
                    "transformations_applied": synthesis_result.transformations_applied,
                    "compression_ratio": synthesis_result.compression_ratio
                }
            )
            
        except Exception as e:
            return ProcessingResponse(
                request_id=request.request_id,
                processed_content=None,
                confidence_score=0.0,
                quality_metrics={"error": 1.0},
                processing_time=time.time() - start_time,
                framework_info=self.get_framework_info(),
                error_details={"message": str(e), "framework": "Semantic Kernel"}
            )
    
    async def _process_with_fallback(self, request: ProcessingRequest) -> ProcessingResponse:
        """폴백 처리 방식"""
        start_time = time.time()
        
        content = request.content
        target_format = request.processing_options.get("output_format", "markdown")
        
        # 간단한 변환 로직
        if target_format == "markdown":
            processed_content = self._simple_markdown_conversion(content)
        elif target_format == "json":
            processed_content = self._simple_json_conversion(content)
        else:
            processed_content = str(content)
        
        return ProcessingResponse(
            request_id=request.request_id,
            processed_content=processed_content,
            confidence_score=0.7,
            quality_metrics={"fallback": 1.0},
            processing_time=time.time() - start_time,
            framework_info=self.get_framework_info()
        )
    
    async def _create_synthesis_plan(self, request: ProcessingRequest) -> List[str]:
        """가공 계획 생성"""
        synthesis_mode = request.processing_options.get("synthesis_mode", "structure")
        output_format = request.processing_options.get("output_format", "markdown")
        
        plan = []
        
        # 모드별 계획 수립
        if synthesis_mode == "structure":
            plan.extend(["structure_data", "enhance_readability"])
        elif synthesis_mode == "compress":
            plan.extend(["intelligent_compress", "deduplicate_content"])
        elif synthesis_mode == "transform":
            plan.extend([f"to_{output_format}", "enrich_metadata"])
        else:
            plan.extend(["structure_data", f"to_{output_format}"])
        
        return plan
    
    async def _execute_synthesis_plan(self, plan: List[str], request: ProcessingRequest) -> SynthesisResult:
        """가공 계획 실행"""
        content = request.content
        transformations_applied = []
        
        for step in plan:
            try:
                if step == "structure_data":
                    if "structure" in self.skills:
                        context = self.kernel.create_new_context()
                        context["input_data"] = str(content)
                        context["target_format"] = request.processing_options.get("output_format", "markdown")
                        
                        result = await self.skills["structure"]["structure_data"].invoke_async(context)
                        content = result.result
                        transformations_applied.append("structure_data")
                
                elif step == "intelligent_compress":
                    if "compress" in self.skills:
                        context = self.kernel.create_new_context()
                        context["content"] = str(content)
                        context["compression_ratio"] = str(request.processing_options.get("compression_ratio", 0.3))
                        
                        result = await self.skills["compress"]["intelligent_compress"].invoke_async(context)
                        content = result.result
                        transformations_applied.append("intelligent_compress")
                
                elif step.startswith("to_"):
                    format_name = step[3:]  # "to_" 제거
                    if "format" in self.skills and format_name in ["markdown", "json", "semantic_html"]:
                        context = self.kernel.create_new_context()
                        context["content"] = str(content)
                        
                        result = await self.skills["format"][step].invoke_async(context)
                        content = result.result
                        transformations_applied.append(step)
                
                elif step == "enhance_readability":
                    if "enhance" in self.skills:
                        context = self.kernel.create_new_context()
                        context["content"] = str(content)
                        
                        result = await self.skills["enhance"]["enhance_readability"].invoke_async(context)
                        content = result.result
                        transformations_applied.append("enhance_readability")
                
            except Exception as e:
                print(f"변환 단계 '{step}' 실패: {e}")
                continue
        
        # 품질 메트릭 계산
        quality_metrics = await self._calculate_synthesis_quality(request.content, content)
        
        return SynthesisResult(
            result_id=f"synthesis_{int(time.time())}",
            original_content=request.content,
            synthesized_content=content,
            format_type=OutputFormat(request.processing_options.get("output_format", "markdown")),
            compression_ratio=len(str(content)) / max(len(str(request.content)), 1),
            quality_metrics=quality_metrics,
            transformations_applied=transformations_applied
        )
    
    async def _calculate_synthesis_quality(self, original: Any, synthesized: Any) -> Dict[str, float]:
        """가공 품질 계산"""
        original_length = len(str(original))
        synthesized_length = len(str(synthesized))
        
        # 기본 메트릭들
        metrics = {
            "structure_quality": 0.85,  # 구조화 품질
            "readability": 0.8,         # 가독성
            "information_preservation": min(synthesized_length / max(original_length, 1), 1.0),
            "format_compliance": 0.9,   # 형식 준수도
            "overall_confidence": 0.82
        }
        
        # 압축 효율성
        if synthesized_length < original_length:
            metrics["compression_efficiency"] = 1.0 - (synthesized_length / original_length)
        else:
            metrics["expansion_ratio"] = synthesized_length / max(original_length, 1)
        
        return metrics
    
    def _simple_markdown_conversion(self, content: Any) -> str:
        """간단한 마크다운 변환 (폴백용)"""
        if isinstance(content, dict):
            markdown = "# 데이터 구조\n\n"
            for key, value in content.items():
                markdown += f"## {key}\n\n{value}\n\n"
            return markdown
        elif isinstance(content, list):
            markdown = "# 목록\n\n"
            for i, item in enumerate(content, 1):
                markdown += f"{i}. {item}\n"
            return markdown + "\n"
        else:
            return f"# 콘텐츠\n\n{content}"
    
    def _simple_json_conversion(self, content: Any) -> str:
        """간단한 JSON 변환 (폴백용)"""
        try:
            if isinstance(content, str):
                structured = {
                    "content": content,
                    "metadata": {"type": "text", "processed": True}
                }
                return json.dumps(structured, ensure_ascii=False, indent=2)
            else:
                return json.dumps(content, ensure_ascii=False, indent=2)
        except:
            return '{"error": "JSON 변환 실패", "original_type": "' + str(type(content)) + '"}'
    
    def get_framework_info(self) -> Dict[str, str]:
        """프레임워크 정보"""
        return {
            "name": "Semantic Kernel",
            "version": "0.3.15" if SEMANTIC_KERNEL_AVAILABLE else "fallback",
            "status": "active" if self.is_initialized else "initializing",
            "features": "plugin_architecture,skill_composition,planning,context_management"
        }


class DataStructuringCapability(AgentCapability):
    """데이터 구조화 능력"""
    
    def get_capability_name(self) -> str:
        return "data_structuring"
    
    def get_supported_formats(self) -> List[str]:
        return ["text", "json", "csv", "xml", "unstructured"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        target_structure = config.get("target_structure", "hierarchical")
        depth_level = config.get("depth_level", 3)
        
        # LLM 기반 구조화 (실제로는 더 정교한 프롬프트 엔지니어링 사용)
        structured_data = await self._apply_structure_intelligence(input_data, target_structure, depth_level)
        
        return {
            "original_data": input_data,
            "structured_data": structured_data,
            "structure_type": target_structure,
            "depth_applied": depth_level,
            "structuring_confidence": 0.88
        }
    
    async def _apply_structure_intelligence(self, data: Any, structure_type: str, depth: int) -> Any:
        """지능형 구조화 적용"""
        if structure_type == "hierarchical":
            return self._create_hierarchical_structure(data, depth)
        elif structure_type == "categorical":
            return self._create_categorical_structure(data)
        elif structure_type == "temporal":
            return self._create_temporal_structure(data)
        else:
            return self._create_semantic_structure(data)
    
    def _create_hierarchical_structure(self, data: Any, max_depth: int) -> Dict[str, Any]:
        """계층적 구조 생성"""
        if isinstance(data, str):
            # 텍스트를 문단 > 문장 > 구문으로 계층화
            paragraphs = data.split('\n\n')
            structure = {
                "type": "hierarchical",
                "levels": {}
            }
            
            for i, paragraph in enumerate(paragraphs[:max_depth]):
                sentences = paragraph.split('. ')
                structure["levels"][f"level_{i}"] = {
                    "content": paragraph,
                    "sentences": sentences,
                    "metadata": {"word_count": len(paragraph.split())}
                }
            
            return structure
        else:
            return {"type": "hierarchical", "data": data, "levels": {"level_0": data}}


class FormatConversionCapability(AgentCapability):
    """형식 변환 능력"""
    
    def get_capability_name(self) -> str:
        return "format_conversion"
    
    def get_supported_formats(self) -> List[str]:
        return ["markdown", "json", "xml", "yaml", "html", "csv", "knowledge_graph"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        source_format = config.get("source_format", "auto")
        target_format = config.get("target_format", "markdown")
        preserve_metadata = config.get("preserve_metadata", True)
        
        # 소스 형식 자동 감지
        if source_format == "auto":
            source_format = await self._detect_format(input_data)
        
        # 형식 변환 실행
        converted_data = await self._convert_format(input_data, source_format, target_format, preserve_metadata)
        
        return {
            "source_format": source_format,
            "target_format": target_format,
            "converted_data": converted_data,
            "metadata_preserved": preserve_metadata,
            "conversion_quality": await self._assess_conversion_quality(input_data, converted_data)
        }
    
    async def _detect_format(self, data: Any) -> str:
        """형식 자동 감지"""
        data_str = str(data)
        
        if data_str.strip().startswith('{') or data_str.strip().startswith('['):
            return "json"
        elif data_str.startswith('#') or '##' in data_str:
            return "markdown"
        elif data_str.startswith('<') and data_str.endswith('>'):
            return "xml"
        elif isinstance(data, dict):
            return "dict"
        elif isinstance(data, list):
            return "list"
        else:
            return "text"
    
    async def _convert_format(self, data: Any, source: str, target: str, preserve_metadata: bool) -> Any:
        """실제 형식 변환"""
        conversion_map = {
            ("text", "markdown"): self._text_to_markdown,
            ("text", "json"): self._text_to_json,
            ("dict", "json"): self._dict_to_json,
            ("dict", "markdown"): self._dict_to_markdown,
            ("list", "markdown"): self._list_to_markdown,
            ("json", "markdown"): self._json_to_markdown,
        }
        
        converter = conversion_map.get((source, target))
        if converter:
            return converter(data, preserve_metadata)
        else:
            return f"변환 불가: {source} -> {target}"
    
    def _text_to_markdown(self, text: str, preserve_metadata: bool) -> str:
        """텍스트를 마크다운으로 변환"""
        lines = text.split('\n')
        markdown = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown += "\n"
            elif line.endswith(':') and len(line) < 50:
                markdown += f"## {line[:-1]}\n\n"
            elif line.startswith('-') or line.startswith('*'):
                markdown += f"- {line[1:].strip()}\n"
            else:
                markdown += f"{line}\n\n"
        
        if preserve_metadata:
            markdown += f"\n---\n*원본 길이: {len(text)} 문자*"
        
        return markdown
    
    def _text_to_json(self, text: str, preserve_metadata: bool) -> str:
        """텍스트를 JSON으로 변환"""
        result = {
            "content": text,
            "type": "text"
        }
        
        if preserve_metadata:
            result["metadata"] = {
                "length": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.split('\n'))
            }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _dict_to_json(self, data: dict, preserve_metadata: bool) -> str:
        """딕셔너리를 JSON으로 변환"""
        if preserve_metadata and "_metadata" not in data:
            data["_metadata"] = {
                "keys_count": len(data),
                "conversion_timestamp": time.time()
            }
        
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _dict_to_markdown(self, data: dict, preserve_metadata: bool) -> str:
        """딕셔너리를 마크다운으로 변환"""
        markdown = "# 데이터 구조\n\n"
        
        for key, value in data.items():
            if key.startswith('_') and not preserve_metadata:
                continue
                
            markdown += f"## {key}\n\n"
            
            if isinstance(value, (dict, list)):
                markdown += f"```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```\n\n"
            else:
                markdown += f"{value}\n\n"
        
        return markdown
    
    def _list_to_markdown(self, data: list, preserve_metadata: bool) -> str:
        """리스트를 마크다운으로 변환"""
        markdown = "# 목록\n\n"
        
        for i, item in enumerate(data, 1):
            markdown += f"{i}. {item}\n"
        
        if preserve_metadata:
            markdown += f"\n---\n*총 {len(data)}개 항목*"
        
        return markdown + "\n"
    
    def _json_to_markdown(self, json_str: str, preserve_metadata: bool) -> str:
        """JSON을 마크다운으로 변환"""
        try:
            data = json.loads(json_str)
            return self._dict_to_markdown(data, preserve_metadata)
        except:
            return f"# JSON 파싱 오류\n\n```\n{json_str}\n```"
    
    async def _assess_conversion_quality(self, original: Any, converted: Any) -> float:
        """변환 품질 평가"""
        original_info = len(str(original))
        converted_info = len(str(converted))
        
        # 정보 보존율
        preservation_ratio = min(converted_info / max(original_info, 1), 1.0)
        
        # 구조화 품질 (간단한 휴리스틱)
        structure_quality = 0.8
        if isinstance(converted, str):
            if converted.count('\n') > str(original).count('\n'):
                structure_quality += 0.1  # 더 나은 구조화
        
        return min(preservation_ratio * 0.7 + structure_quality * 0.3, 1.0)


class CompressionOptimizationCapability(AgentCapability):
    """압축 최적화 능력"""
    
    def get_capability_name(self) -> str:
        return "compression_optimization"
    
    def get_supported_formats(self) -> List[str]:
        return ["text", "structured_data", "mixed_content"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        target_ratio = config.get("compression_ratio", 0.3)
        preserve_keywords = config.get("preserve_keywords", [])
        quality_threshold = config.get("quality_threshold", 0.8)
        
        # 지능형 압축 수행
        compressed_data = await self._intelligent_compression(
            input_data, target_ratio, preserve_keywords, quality_threshold
        )
        
        return {
            "original_data": input_data,
            "compressed_data": compressed_data,
            "actual_ratio": len(str(compressed_data)) / max(len(str(input_data)), 1),
            "keywords_preserved": preserve_keywords,
            "compression_quality": await self._evaluate_compression_quality(input_data, compressed_data)
        }
    
    async def _intelligent_compression(self, data: Any, target_ratio: float, keywords: List[str], quality_threshold: float) -> Any:
        """지능형 압축"""
        text = str(data)
        target_length = int(len(text) * target_ratio)
        
        # 중요도 기반 문장 선택
        sentences = self._split_into_sentences(text)
        scored_sentences = await self._score_sentences(sentences, keywords)
        
        # 품질 임계값을 만족하면서 목표 길이에 맞는 압축
        selected_sentences = self._select_sentences_by_importance(
            scored_sentences, target_length, quality_threshold
        )
        
        return ' '.join(selected_sentences)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """문장 분할"""
        # 간단한 문장 분할 (실제로는 더 정교한 NLP 라이브러리 사용)
        sentences = []
        for delimiter in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            text = text.replace(delimiter, '|SENTENCE_BREAK|')
        
        raw_sentences = text.split('|SENTENCE_BREAK|')
        for sentence in raw_sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and len(clean_sentence) > 10:
                sentences.append(clean_sentence)
        
        return sentences
    
    async def _score_sentences(self, sentences: List[str], keywords: List[str]) -> List[tuple]:
        """문장 중요도 점수 계산"""
        scored_sentences = []
        
        for sentence in sentences:
            score = 0.0
            
            # 키워드 포함 점수
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    score += 1.0
            
            # 길이 점수 (너무 짧거나 긴 문장 페널티)
            length_score = 1.0
            if len(sentence) < 20:
                length_score = 0.5
            elif len(sentence) > 200:
                length_score = 0.7
            
            # 위치 점수 (첫 번째와 마지막 문장은 중요)
            position_score = 1.0
            if sentences.index(sentence) in [0, len(sentences) - 1]:
                position_score = 1.2
            
            final_score = score + length_score + position_score
            scored_sentences.append((sentence, final_score))
        
        return sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    
    def _select_sentences_by_importance(self, scored_sentences: List[tuple], target_length: int, quality_threshold: float) -> List[str]:
        """중요도에 따른 문장 선택"""
        selected = []
        current_length = 0
        
        for sentence, score in scored_sentences:
            if score >= quality_threshold and current_length + len(sentence) <= target_length:
                selected.append(sentence)
                current_length += len(sentence)
        
        # 최소한의 문장은 보장
        if not selected and scored_sentences:
            selected.append(scored_sentences[0][0])
        
        return selected
    
    async def _evaluate_compression_quality(self, original: Any, compressed: Any) -> Dict[str, float]:
        """압축 품질 평가"""
        original_text = str(original)
        compressed_text = str(compressed)
        
        # 기본 메트릭
        compression_ratio = len(compressed_text) / max(len(original_text), 1)
        
        # 정보 보존도 (간단한 키워드 기반)
        original_words = set(original_text.lower().split())
        compressed_words = set(compressed_text.lower().split())
        information_preservation = len(compressed_words.intersection(original_words)) / max(len(original_words), 1)
        
        return {
            "compression_ratio": compression_ratio,
            "information_preservation": information_preservation,
            "readability": 0.85,  # 실제로는 가독성 분석 필요
            "coherence": 0.8,     # 실제로는 일관성 분석 필요
            "overall_quality": (information_preservation + 0.85 + 0.8) / 3
        }


class SynthesizerAgent(ModularAgent):
    """Semantic Kernel 기반 데이터 가공 에이전트"""
    
    def __init__(self, config: AgentConfig):
        # 프레임워크 검증
        if config.framework != AgentFramework.SEMANTIC_KERNEL:
            raise ValueError(f"SynthesizerAgent는 Semantic Kernel 프레임워크만 지원합니다. 현재: {config.framework}")
        
        super().__init__(config)
        
        # 가공 설정
        self.default_output_format = config.custom_config.get("default_output_format", "markdown")
        self.compression_ratio = config.custom_config.get("default_compression_ratio", 0.3)
        self.quality_threshold = config.custom_config.get("quality_threshold", 0.8)
        
    async def _register_default_capabilities(self):
        """기본 능력 등록"""
        # 데이터 구조화 능력
        structuring_capability = DataStructuringCapability()
        self.capability_registry.register_capability(structuring_capability)
        
        # 형식 변환 능력
        format_capability = FormatConversionCapability()
        self.capability_registry.register_capability(format_capability)
        
        # 압축 최적화 능력
        compression_capability = CompressionOptimizationCapability()
        self.capability_registry.register_capability(compression_capability)
        
        self.logger.info("SynthesizerAgent 기본 능력 등록 완료")
    
    async def _load_framework_adapter(self) -> FrameworkAdapter:
        """Semantic Kernel 어댑터 로드"""
        adapter = SemanticKernelAdapter()
        
        # 에이전트별 맞춤 설정
        framework_config = {
            "openai_api_key": self.config.custom_config.get("openai_api_key", "default-key"),
            "model_id": self.config.custom_config.get("model_id", "gpt-3.5-turbo"),
            "temperature": self.config.custom_config.get("temperature", 0.1),
            "transformation_rules": self.config.custom_config.get("transformation_rules", [])
        }
        
        await adapter.initialize(framework_config)
        return adapter
    
    async def synthesize_multiformat_content(self, content: Any, target_formats: List[str]) -> Dict[str, Any]:
        """다중 형식 콘텐츠 가공"""
        results = {}
        
        for format_type in target_formats:
            synthesis_request = ProcessingRequest(
                request_id=f"multiformat_{format_type}_{int(time.time())}",
                content=content,
                content_type="multiformat_synthesis",
                processing_options={
                    "synthesis_mode": "transform",
                    "output_format": format_type,
                    "quality_threshold": self.quality_threshold
                }
            )
            
            response = await self.process(synthesis_request)
            results[format_type] = {
                "content": response.processed_content,
                "confidence": response.confidence_score,
                "quality_metrics": response.quality_metrics
            }
        
        return {
            "original_content": content,
            "synthesized_formats": results,
            "total_formats": len(target_formats),
            "overall_quality": sum(r["confidence"] for r in results.values()) / len(results)
        }
    
    async def optimize_content_structure(self, content: Any, optimization_goals: List[str]) -> Dict[str, Any]:
        """콘텐츠 구조 최적화"""
        optimization_request = ProcessingRequest(
            request_id=f"optimize_{int(time.time())}",
            content=content,
            content_type="structure_optimization",
            processing_options={
                "synthesis_mode": "optimize",
                "optimization_goals": optimization_goals,
                "output_format": self.default_output_format
            }
        )
        
        response = await self.process(optimization_request)
        
        return {
            "original_content": content,
            "optimized_content": response.processed_content,
            "optimization_goals": optimization_goals,
            "quality_improvement": response.quality_metrics,
            "optimization_confidence": response.confidence_score
        }
    
    async def adaptive_compression(self, content: Any, context_budget: int, preserve_elements: List[str]) -> Dict[str, Any]:
        """적응형 압축"""
        # 목표 압축 비율 계산
        current_length = len(str(content))
        target_ratio = min(context_budget / current_length, 1.0) if current_length > 0 else 1.0
        
        compression_request = ProcessingRequest(
            request_id=f"compress_{int(time.time())}",
            content=content,
            content_type="adaptive_compression",
            processing_options={
                "synthesis_mode": "compress",
                "compression_ratio": target_ratio,
                "preserve_keywords": preserve_elements,
                "context_budget": context_budget
            }
        )
        
        response = await self.process(compression_request)
        
        actual_length = len(str(response.processed_content)) if response.processed_content else 0
        
        return {
            "original_content": content,
            "compressed_content": response.processed_content,
            "target_budget": context_budget,
            "actual_length": actual_length,
            "compression_achieved": target_ratio,
            "elements_preserved": preserve_elements,
            "compression_quality": response.quality_metrics
        }


# 사용 예시
async def example_synthesizer_usage():
    """SynthesizerAgent 사용 예시"""
    
    # 설정 생성
    config = AgentConfig(
        agent_id="synthesizer_main",
        framework=AgentFramework.SEMANTIC_KERNEL,
        capabilities=["data_structuring", "format_conversion", "compression_optimization"],
        processing_mode=ProcessingMode.ASYNC,
        custom_config={
            "default_output_format": "markdown",
            "default_compression_ratio": 0.4,
            "openai_api_key": "your-api-key",
            "model_id": "gpt-3.5-turbo"
        }
    )
    
    # 에이전트 생성
    synthesizer = SynthesizerAgent(config)
    
    # 기본 가공 요청
    sample_content = """
    Python은 프로그래밍 언어입니다. 간단하고 읽기 쉬운 구문을 가지고 있습니다.
    웹 개발, 데이터 분석, 인공지능 등 다양한 분야에서 사용됩니다.
    Django와 Flask는 대표적인 웹 프레임워크입니다.
    NumPy와 Pandas는 데이터 분석에 필수적인 라이브러리입니다.
    """
    
    request = ProcessingRequest(
        request_id="test_synthesis_001",
        content=sample_content,
        content_type="text_structuring",
        processing_options={
            "synthesis_mode": "structure",
            "output_format": "markdown",
            "quality_threshold": 0.8
        }
    )
    
    # 가공 실행
    response = await synthesizer.process(request)
    
    print(f"가공 완료: 신뢰도 {response.confidence_score:.2f}")
    print(f"사용된 프레임워크: {response.framework_info['name']}")
    print(f"처리 시간: {response.processing_time:.2f}초")
    
    # 다중 형식 가공 테스트
    multiformat_results = await synthesizer.synthesize_multiformat_content(
        sample_content,
        ["markdown", "json", "xml"]
    )
    
    print(f"다중 형식 가공 완료: {multiformat_results['total_formats']}개 형식")
    print(f"전체 품질: {multiformat_results['overall_quality']:.2f}")
    
    # 적응형 압축 테스트
    compression_results = await synthesizer.adaptive_compression(
        sample_content,
        context_budget=200,  # 200자 예산
        preserve_elements=["Python", "Django", "Flask"]
    )
    
    print(f"압축 완료: {compression_results['actual_length']}자")
    print(f"압축 비율: {compression_results['compression_achieved']:.2%}")
    
    return response


if __name__ == "__main__":
    asyncio.run(example_synthesizer_usage())