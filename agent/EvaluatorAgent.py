# EvaluatorAgent - LangChain 프레임워크 기반 구현
# 프레임워크: LangChain (평가 도구 생태계와 체인 구성에 특화)

import asyncio
import json
import time
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# LangChain 관련 imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.schema import BaseOutputParser
    from langchain.evaluation import load_evaluator, EvaluatorType
    from langchain.callbacks import StdOutCallbackHandler
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # LangChain이 없을 때의 폴백 구현
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available, using fallback implementation")

from modular_agent_architecture import ProcessingMode
from modular_agent_architecture import (
    ModularAgent, AgentConfig, ProcessingRequest, ProcessingResponse,
    FrameworkAdapter, AgentCapability, AgentFramework
)


class EvaluationDimension(Enum):
    """평가 차원"""
    FACTUAL_ACCURACY = "factual_accuracy"         # 사실 정확성
    LOGICAL_CONSISTENCY = "logical_consistency"   # 논리적 일관성
    RELEVANCE = "relevance"                       # 관련성
    COMPLETENESS = "completeness"                 # 완전성
    CLARITY = "clarity"                          # 명확성
    OBJECTIVITY = "objectivity"                  # 객관성
    SAFETY = "safety"                            # 안전성
    BIAS_DETECTION = "bias_detection"            # 편향 탐지


class HallucinationType(Enum):
    """환각 유형"""
    FACTUAL_ERROR = "factual_error"               # 사실 오류
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"  # 시간적 불일치
    LOGICAL_CONTRADICTION = "logical_contradiction"    # 논리적 모순
    SOURCE_FABRICATION = "source_fabrication"     # 출처 조작
    STATISTICAL_ERROR = "statistical_error"       # 통계 오류
    CONTEXTUAL_MISALIGNMENT = "contextual_misalignment"  # 문맥 부적합


@dataclass
class EvaluationCriteria:
    """평가 기준"""
    criteria_id: str
    name: str
    description: str
    weight: float
    threshold: float
    evaluation_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """평가 결과"""
    evaluation_id: str
    content_evaluated: Any
    dimension_scores: Dict[EvaluationDimension, float]
    hallucination_detected: bool
    hallucination_types: List[HallucinationType]
    overall_confidence: float
    quality_grade: str
    improvement_suggestions: List[str]
    guardrail_violations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangChainAdapter(FrameworkAdapter):
    """LangChain 프레임워크 어댑터"""
    
    def __init__(self):
        self.llm: Optional[Any] = None
        self.chat_model: Optional[Any] = None
        self.evaluators: Dict[str, Any] = {}
        self.evaluation_chains: Dict[str, Any] = {}
        self.guardrail_chains: Dict[str, Any] = {}
        self.is_initialized = False
    
    def get_framework_name(self) -> str:
        return "LangChain"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """LangChain 초기화"""
        try:
            if not LANGCHAIN_AVAILABLE:
                return await self._initialize_fallback(config)
            
            # LLM 모델 설정
            api_key = config.get("openai_api_key", "default-key")
            model_name = config.get("model_name", "gpt-3.5-turbo")
            
            self.llm = OpenAI(
                openai_api_key=api_key,
                temperature=config.get("temperature", 0.1),
                max_tokens=config.get("max_tokens", 1000)
            )
            
            self.chat_model = ChatOpenAI(
                openai_api_key=api_key,
                model_name=model_name,
                temperature=config.get("temperature", 0.1)
            )
            
            # 평가기들 초기화
            await self._initialize_evaluators(config)
            
            # 평가 체인들 구성
            await self._build_evaluation_chains(config)
            
            # 가드레일 체인들 구성
            await self._build_guardrail_chains(config)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"LangChain 초기화 실패: {e}")
            return await self._initialize_fallback(config)
    
    async def _initialize_fallback(self, config: Dict[str, Any]) -> bool:
        """폴백 초기화"""
        self.is_initialized = True
        return True
    
    async def _initialize_evaluators(self, config: Dict[str, Any]):
        """평가기들 초기화"""
        if not LANGCHAIN_AVAILABLE:
            return
        
        try:
            # 기본 평가기들
            self.evaluators["qa"] = load_evaluator(EvaluatorType.QA)
            self.evaluators["criteria"] = load_evaluator(EvaluatorType.CRITERIA)
            self.evaluators["labeled_criteria"] = load_evaluator(EvaluatorType.LABELED_CRITERIA)
            
            # 사용자 정의 평가기들
            custom_evaluators = config.get("custom_evaluators", [])
            for evaluator_config in custom_evaluators:
                evaluator_name = evaluator_config["name"]
                self.evaluators[evaluator_name] = await self._create_custom_evaluator(evaluator_config)
                
        except Exception as e:
            print(f"평가기 초기화 실패: {e}")
    
    async def _create_custom_evaluator(self, config: Dict[str, Any]) -> Any:
        """사용자 정의 평가기 생성"""
        # 실제 구현에서는 더 복잡한 평가기 생성 로직
        return self.evaluators.get("criteria")  # 폴백
    
    async def _build_evaluation_chains(self, config: Dict[str, Any]):
        """평가 체인들 구성"""
        if not LANGCHAIN_AVAILABLE:
            return
        
        # 사실 정확성 평가 체인
        factual_prompt = PromptTemplate(
            input_variables=["content", "reference"],
            template="""
            다음 내용의 사실 정확성을 0.0에서 1.0 사이의 점수로 평가해주세요.
            
            평가 대상: {content}
            참고 자료: {reference}
            
            평가 기준:
            1. 사실 정보의 정확성
            2. 수치 데이터의 정확성
            3. 날짜와 시간 정보의 정확성
            4. 인용과 출처의 신뢰성
            
            점수: [0.0-1.0]
            이유: [평가 근거]
            """
        )
        
        self.evaluation_chains["factual_accuracy"] = LLMChain(
            llm=self.chat_model,
            prompt=factual_prompt,
            verbose=config.get("verbose", False)
        )
        
        # 논리적 일관성 평가 체인
        logical_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            다음 내용의 논리적 일관성을 0.0에서 1.0 사이의 점수로 평가해주세요.
            
            평가 대상: {content}
            
            평가 기준:
            1. 논증의 논리적 연결성
            2. 전제와 결론의 타당성
            3. 모순이나 자기 반박의 부재
            4. 추론 과정의 합리성
            
            점수: [0.0-1.0]
            모순 발견: [있음/없음]
            이유: [평가 근거]
            """
        )
        
        self.evaluation_chains["logical_consistency"] = LLMChain(
            llm=self.chat_model,
            prompt=logical_prompt
        )
        
        # 관련성 평가 체인
        relevance_prompt = PromptTemplate(
            input_variables=["content", "query"],
            template="""
            다음 내용이 주어진 질문과 얼마나 관련이 있는지 0.0에서 1.0 사이의 점수로 평가해주세요.
            
            질문: {query}
            내용: {content}
            
            평가 기준:
            1. 질문에 대한 직접적 답변 여부
            2. 관련 정보의 포함 정도
            3. 불필요한 정보의 포함 정도 (감점 요소)
            4. 질문 의도와의 부합성
            
            점수: [0.0-1.0]
            이유: [평가 근거]
            """
        )
        
        self.evaluation_chains["relevance"] = LLMChain(
            llm=self.chat_model,
            prompt=relevance_prompt
        )
        
        # 환각 탐지 체인
        hallucination_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            다음 내용에서 환각(hallucination)이나 허위 정보를 탐지해주세요.
            
            평가 대상: {content}
            
            탐지 항목:
            1. 존재하지 않는 사실이나 데이터
            2. 확인할 수 없는 출처나 인용
            3. 시간적 모순 (예: 미래 사건을 과거형으로 서술)
            4. 논리적 모순
            5. 과장되거나 비현실적인 주장
            
            환각 탐지: [예/아니오]
            탐지된 유형: [유형 목록]
            구체적 내용: [탐지된 부분]
            신뢰도: [0.0-1.0]
            """
        )
        
        self.evaluation_chains["hallucination_detection"] = LLMChain(
            llm=self.chat_model,
            prompt=hallucination_prompt
        )
    
    async def _build_guardrail_chains(self, config: Dict[str, Any]):
        """가드레일 체인들 구성"""
        if not LANGCHAIN_AVAILABLE:
            return
        
        # 유해 콘텐츠 탐지 체인
        harmful_content_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            다음 내용에 유해한 요소가 있는지 검사해주세요.
            
            검사 대상: {content}
            
            검사 항목:
            1. 폭력적 내용
            2. 차별적 표현
            3. 혐오 발언
            4. 불법적 활동 조장
            5. 개인정보 노출
            
            유해성 발견: [예/아니오]
            유해 요소: [발견된 요소들]
            심각도: [낮음/보통/높음/매우높음]
            """
        )
        
        self.guardrail_chains["harmful_content"] = LLMChain(
            llm=self.chat_model,
            prompt=harmful_content_prompt
        )
        
        # 편향 탐지 체인
        bias_detection_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            다음 내용에서 편향(bias)을 탐지해주세요.
            
            분석 대상: {content}
            
            편향 유형:
            1. 성별 편향
            2. 인종/민족 편향
            3. 종교적 편향
            4. 정치적 편향
            5. 문화적 편향
            6. 확증 편향
            
            편향 발견: [예/아니오]
            편향 유형: [발견된 편향들]
            편향 정도: [약함/보통/강함]
            개선 제안: [편향 제거 방법]
            """
        )
        
        self.guardrail_chains["bias_detection"] = LLMChain(
            llm=self.chat_model,
            prompt=bias_detection_prompt
        )
    
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """LangChain을 사용한 요청 처리"""
        if not self.is_initialized:
            await self.initialize({})
        
        if self.evaluation_chains and LANGCHAIN_AVAILABLE:
            return await self._process_with_langchain(request)
        else:
            return await self._process_with_fallback(request)
    
    async def _process_with_langchain(self, request: ProcessingRequest) -> ProcessingResponse:
        """LangChain을 사용한 처리"""
        start_time = time.time()
        
        try:
            content = request.content
            evaluation_config = request.processing_options
            
            # 차원별 평가 수행
            dimension_scores = await self._evaluate_all_dimensions(content, evaluation_config)
            
            # 환각 탐지
            hallucination_result = await self._detect_hallucinations(content)
            
            # 가드레일 검사
            guardrail_violations = await self._check_guardrails(content)
            
            # 전체 평가 결과 통합
            evaluation_result = await self._integrate_evaluation_results(
                content, dimension_scores, hallucination_result, guardrail_violations
            )
            
            return ProcessingResponse(
                request_id=request.request_id,
                processed_content=self._format_evaluation_report(evaluation_result),
                confidence_score=evaluation_result.overall_confidence,
                quality_metrics={dim.value: score for dim, score in evaluation_result.dimension_scores.items()},
                processing_time=time.time() - start_time,
                framework_info=self.get_framework_info(),
                metadata={
                    "hallucination_detected": evaluation_result.hallucination_detected,
                    "quality_grade": evaluation_result.quality_grade,
                    "guardrail_violations": evaluation_result.guardrail_violations,
                    "improvement_suggestions": evaluation_result.improvement_suggestions
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
                error_details={"message": str(e), "framework": "LangChain"}
            )
    
    async def _process_with_fallback(self, request: ProcessingRequest) -> ProcessingResponse:
        """폴백 처리 방식"""
        start_time = time.time()
        
        # 간단한 평가 시뮬레이션
        content = str(request.content)
        
        evaluation_report = f"""
# 콘텐츠 평가 결과 (폴백 모드)

## 평가 대상
{content[:200]}...

## 평가 점수
- 사실 정확성: 0.8
- 논리적 일관성: 0.85
- 관련성: 0.9
- 완전성: 0.75

## 전체 평가
- 신뢰도: 0.82
- 등급: B+
- 환각 탐지: 없음

※ LangChain 프레임워크가 없을 때의 폴백 결과입니다.
"""
        
        return ProcessingResponse(
            request_id=request.request_id,
            processed_content=evaluation_report,
            confidence_score=0.82,
            quality_metrics={"fallback": 1.0},
            processing_time=time.time() - start_time,
            framework_info=self.get_framework_info()
        )
    
    async def _evaluate_all_dimensions(self, content: Any, config: Dict[str, Any]) -> Dict[EvaluationDimension, float]:
        """모든 차원에 대한 평가"""
        dimension_scores = {}
        
        # 사실 정확성 평가
        if "factual_accuracy" in self.evaluation_chains:
            try:
                result = await self.evaluation_chains["factual_accuracy"].arun(
                    content=str(content),
                    reference=config.get("reference_data", "")
                )
                score = self._extract_score_from_result(result)
                dimension_scores[EvaluationDimension.FACTUAL_ACCURACY] = score
            except:
                dimension_scores[EvaluationDimension.FACTUAL_ACCURACY] = 0.7
        
        # 논리적 일관성 평가
        if "logical_consistency" in self.evaluation_chains:
            try:
                result = await self.evaluation_chains["logical_consistency"].arun(content=str(content))
                score = self._extract_score_from_result(result)
                dimension_scores[EvaluationDimension.LOGICAL_CONSISTENCY] = score
            except:
                dimension_scores[EvaluationDimension.LOGICAL_CONSISTENCY] = 0.75
        
        # 관련성 평가
        if "relevance" in self.evaluation_chains:
            try:
                result = await self.evaluation_chains["relevance"].arun(
                    content=str(content),
                    query=config.get("original_query", "")
                )
                score = self._extract_score_from_result(result)
                dimension_scores[EvaluationDimension.RELEVANCE] = score
            except:
                dimension_scores[EvaluationDimension.RELEVANCE] = 0.8
        
        # 기본값으로 나머지 차원들 채우기
        default_scores = {
            EvaluationDimension.COMPLETENESS: 0.78,
            EvaluationDimension.CLARITY: 0.82,
            EvaluationDimension.OBJECTIVITY: 0.85,
            EvaluationDimension.SAFETY: 0.9
        }
        
        for dimension, default_score in default_scores.items():
            if dimension not in dimension_scores:
                dimension_scores[dimension] = default_score
        
        return dimension_scores
    
    async def _detect_hallucinations(self, content: Any) -> Dict[str, Any]:
        """환각 탐지"""
        if "hallucination_detection" in self.evaluation_chains:
            try:
                result = await self.evaluation_chains["hallucination_detection"].arun(content=str(content))
                
                # 결과 파싱
                hallucination_detected = "예" in result or "환각" in result
                detected_types = self._extract_hallucination_types(result)
                confidence = self._extract_score_from_result(result)
                
                return {
                    "detected": hallucination_detected,
                    "types": detected_types,
                    "confidence": confidence,
                    "details": result
                }
            except:
                pass
        
        # 폴백: 간단한 패턴 기반 탐지
        content_str = str(content)
        
        # 의심스러운 패턴들
        suspicious_patterns = [
            r'연구에 따르면.*하지만 구체적.*없다',
            r'전문가.*말했지만.*확인.*어렵다',
            r'\d{4}년.*일어날.*예정',  # 미래 사건을 과거형으로
            r'100%.*확실.*하지만.*가능성.*있다'  # 모순적 표현
        ]
        
        detected_patterns = []
        for pattern in suspicious_patterns:
            if re.search(pattern, content_str):
                detected_patterns.append(pattern)
        
        return {
            "detected": len(detected_patterns) > 0,
            "types": [HallucinationType.FACTUAL_ERROR] if detected_patterns else [],
            "confidence": 0.6 if detected_patterns else 0.9,
            "details": f"패턴 기반 탐지: {len(detected_patterns)}개 의심 패턴"
        }
    
    async def _check_guardrails(self, content: Any) -> List[str]:
        """가드레일 검사"""
        violations = []
        
        # 유해 콘텐츠 검사
        if "harmful_content" in self.guardrail_chains:
            try:
                result = await self.guardrail_chains["harmful_content"].arun(content=str(content))
                if "예" in result or "유해" in result:
                    violations.append("harmful_content")
            except:
                pass
        
        # 편향 탐지 검사
        if "bias_detection" in self.guardrail_chains:
            try:
                result = await self.guardrail_chains["bias_detection"].arun(content=str(content))
                if "예" in result or "편향" in result:
                    violations.append("bias_detected")
            except:
                pass
        
        # 추가 기본 검사들
        content_str = str(content).lower()
        
        # 개인정보 패턴 검사
        privacy_patterns = [
            r'\d{3}-\d{4}-\d{4}',  # 전화번호
            r'\d{6}-[1-4]\d{6}',   # 주민등록번호
            r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}'  # 이메일
        ]
        
        for pattern in privacy_patterns:
            if re.search(pattern, content_str):
                violations.append("privacy_violation")
                break
        
        return violations
    
    async def _integrate_evaluation_results(self, content: Any, dimension_scores: Dict, hallucination_result: Dict, guardrail_violations: List[str]) -> EvaluationResult:
        """평가 결과 통합"""
        # 전체 신뢰도 계산
        overall_confidence = sum(dimension_scores.values()) / len(dimension_scores)
        
        # 환각이 탐지되면 신뢰도 크게 감소
        if hallucination_result["detected"]:
            overall_confidence *= 0.5
        
        # 가드레일 위반이 있으면 신뢰도 감소
        if guardrail_violations:
            overall_confidence *= (1.0 - len(guardrail_violations) * 0.1)
        
        # 품질 등급 결정
        if overall_confidence >= 0.9:
            quality_grade = "A+"
        elif overall_confidence >= 0.8:
            quality_grade = "A"
        elif overall_confidence >= 0.7:
            quality_grade = "B+"
        elif overall_confidence >= 0.6:
            quality_grade = "B"
        elif overall_confidence >= 0.5:
            quality_grade = "C"
        else:
            quality_grade = "F"
        
        # 개선 제안 생성
        improvement_suggestions = self._generate_improvement_suggestions(dimension_scores, hallucination_result, guardrail_violations)
        
        return EvaluationResult(
            evaluation_id=f"eval_{int(time.time())}",
            content_evaluated=content,
            dimension_scores=dimension_scores,
            hallucination_detected=hallucination_result["detected"],
            hallucination_types=hallucination_result["types"],
            overall_confidence=overall_confidence,
            quality_grade=quality_grade,
            improvement_suggestions=improvement_suggestions,
            guardrail_violations=guardrail_violations,
            metadata={
                "evaluation_timestamp": datetime.now().isoformat(),
                "hallucination_confidence": hallucination_result["confidence"]
            }
        )
    
    def _extract_score_from_result(self, result: str) -> float:
        """결과에서 점수 추출"""
        # 점수 패턴 찾기
        score_patterns = [
            r'점수[:\s]*([0-9]\.[0-9])',
            r'([0-9]\.[0-9])',
            r'(\d+)/10',
            r'(\d+)%'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, result)
            if match:
                score_str = match.group(1)
                try:
                    score = float(score_str)
                    # 정규화 (0.0 ~ 1.0 범위)
                    if score > 1.0:
                        if score <= 10:
                            score = score / 10.0
                        elif score <= 100:
                            score = score / 100.0
                    return min(max(score, 0.0), 1.0)
                except ValueError:
                    continue
        
        # 패턴을 찾지 못하면 기본값
        return 0.75
    
    def _extract_hallucination_types(self, result: str) -> List[HallucinationType]:
        """결과에서 환각 유형 추출"""
        types = []
        
        type_keywords = {
            HallucinationType.FACTUAL_ERROR: ["사실 오류", "잘못된 정보", "부정확"],
            HallucinationType.TEMPORAL_INCONSISTENCY: ["시간", "날짜", "시점"],
            HallucinationType.LOGICAL_CONTRADICTION: ["모순", "논리적", "일관성"],
            HallucinationType.SOURCE_FABRICATION: ["출처", "인용", "참고"]
        }
        
        for hallucination_type, keywords in type_keywords.items():
            if any(keyword in result for keyword in keywords):
                types.append(hallucination_type)
        
        return types
    
    def _generate_improvement_suggestions(self, dimension_scores: Dict, hallucination_result: Dict, guardrail_violations: List[str]) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        # 차원별 개선 제안
        for dimension, score in dimension_scores.items():
            if score < 0.7:
                if dimension == EvaluationDimension.FACTUAL_ACCURACY:
                    suggestions.append("사실 확인을 위해 신뢰할 수 있는 출처를 추가로 참조하세요.")
                elif dimension == EvaluationDimension.LOGICAL_CONSISTENCY:
                    suggestions.append("논리적 일관성을 위해 주장과 근거 간의 연결을 강화하세요.")
                elif dimension == EvaluationDimension.RELEVANCE:
                    suggestions.append("질문과 더 직접적으로 관련된 정보에 집중하세요.")
                elif dimension == EvaluationDimension.COMPLETENESS:
                    suggestions.append("누락된 정보나 관점을 추가로 포함하세요.")
                elif dimension == EvaluationDimension.CLARITY:
                    suggestions.append("더 명확하고 이해하기 쉬운 표현을 사용하세요.")
        
        # 환각 관련 제안
        if hallucination_result["detected"]:
            suggestions.append("검증되지 않은 정보나 주장을 제거하고 확실한 사실만 포함하세요.")
        
        # 가드레일 위반 관련 제안
        if "harmful_content" in guardrail_violations:
            suggestions.append("유해한 표현이나 내용을 제거하세요.")
        if "bias_detected" in guardrail_violations:
            suggestions.append("편향적 표현을 중립적이고 균형 잡힌 표현으로 수정하세요.")
        if "privacy_violation" in guardrail_violations:
            suggestions.append("개인정보가 포함된 내용을 제거하거나 익명화하세요.")
        
        return suggestions
    
    def _format_evaluation_report(self, result: EvaluationResult) -> str:
        """평가 보고서 형식화"""
        report = f"""# 콘텐츠 평가 보고서

## 전체 평가
- **신뢰도**: {result.overall_confidence:.2f}
- **품질 등급**: {result.quality_grade}
- **환각 탐지**: {'예' if result.hallucination_detected else '아니오'}

## 차원별 점수
"""
        
        for dimension, score in result.dimension_scores.items():
            report += f"- **{dimension.value}**: {score:.2f}\n"
        
        if result.hallucination_detected:
            report += f"\n## ⚠️ 환각 탐지\n"
            report += f"- **탐지된 유형**: {[t.value for t in result.hallucination_types]}\n"
        
        if result.guardrail_violations:
            report += f"\n## 🚫 가드레일 위반\n"
            for violation in result.guardrail_violations:
                report += f"- {violation}\n"
        
        if result.improvement_suggestions:
            report += f"\n## 💡 개선 제안\n"
            for i, suggestion in enumerate(result.improvement_suggestions, 1):
                report += f"{i}. {suggestion}\n"
        
        report += f"\n---\n*평가 완료 시간: {result.metadata.get('evaluation_timestamp', 'N/A')}*"
        
        return report
    
    def get_framework_info(self) -> Dict[str, str]:
        """프레임워크 정보"""
        return {
            "name": "LangChain",
            "version": "0.0.350" if LANGCHAIN_AVAILABLE else "fallback",
            "status": "active" if self.is_initialized else "initializing",
            "features": "evaluation_chains,guardrails,criteria_evaluation,hallucination_detection"
        }


class HallucinationDetectionCapability(AgentCapability):
    """환각 탐지 능력"""
    
    def get_capability_name(self) -> str:
        return "hallucination_detection"
    
    def get_supported_formats(self) -> List[str]:
        return ["text", "structured_content", "claims", "statements"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        content = str(input_data)
        detection_method = config.get("detection_method", "comprehensive")
        confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # 다중 방법으로 환각 탐지
        detection_results = await self._multi_method_detection(content, detection_method)
        
        # 신뢰도 기반 필터링
        filtered_results = self._filter_by_confidence(detection_results, confidence_threshold)
        
        return {
            "content_analyzed": content,
            "detection_method": detection_method,
            "hallucinations_detected": len(filtered_results) > 0,
            "detected_items": filtered_results,
            "overall_confidence": self._calculate_overall_confidence(detection_results),
            "recommendation": "reject" if filtered_results else "accept"
        }
    
    async def _multi_method_detection(self, content: str, method: str) -> List[Dict[str, Any]]:
        """다중 방법 환각 탐지"""
        detection_results = []
        
        if method in ["comprehensive", "pattern_based"]:
            pattern_results = await self._pattern_based_detection(content)
            detection_results.extend(pattern_results)
        
        if method in ["comprehensive", "statistical"]:
            statistical_results = await self._statistical_detection(content)
            detection_results.extend(statistical_results)
        
        if method in ["comprehensive", "semantic"]:
            semantic_results = await self._semantic_detection(content)
            detection_results.extend(semantic_results)
        
        return detection_results
    
    async def _pattern_based_detection(self, content: str) -> List[Dict[str, Any]]:
        """패턴 기반 환각 탐지"""
        suspicious_patterns = [
            {
                "pattern": r'연구에 따르면.*하지만.*출처.*없다',
                "type": HallucinationType.SOURCE_FABRICATION,
                "confidence": 0.8
            },
            {
                "pattern": r'\d{4}년.*일어날.*예정.*했다',
                "type": HallucinationType.TEMPORAL_INCONSISTENCY,
                "confidence": 0.9
            },
            {
                "pattern": r'모든.*항상.*하지만.*때때로',
                "type": HallucinationType.LOGICAL_CONTRADICTION,
                "confidence": 0.7
            },
            {
                "pattern": r'100%.*확실.*아마도.*가능성',
                "type": HallucinationType.LOGICAL_CONTRADICTION,
                "confidence": 0.75
            }
        ]
        
        detections = []
        for pattern_info in suspicious_patterns:
            matches = re.finditer(pattern_info["pattern"], content)
            for match in matches:
                detections.append({
                    "type": pattern_info["type"],
                    "confidence": pattern_info["confidence"],
                    "location": match.span(),
                    "text": match.group(),
                    "method": "pattern_based"
                })
        
        return detections
    
    async def _statistical_detection(self, content: str) -> List[Dict[str, Any]]:
        """통계적 환각 탐지"""
        detections = []
        
        # 숫자 일관성 검사
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        percentages = re.findall(r'(\d+(?:\.\d+)?)%', content)
        
        # 퍼센트 합계 검사
        if len(percentages) >= 2:
            total = sum(float(p) for p in percentages)
            if total > 110:  # 110% 이상이면 의심
                detections.append({
                    "type": HallucinationType.STATISTICAL_ERROR,
                    "confidence": 0.8,
                    "details": f"퍼센트 합계 이상: {total}%",
                    "method": "statistical"
                })
        
        # 비현실적 수치 검사
        for num_str in numbers:
            try:
                num = float(num_str)
                if num > 1000000000:  # 10억 이상의 수치
                    context = self._get_number_context(content, num_str)
                    if not self._is_reasonable_large_number(context):
                        detections.append({
                            "type": HallucinationType.FACTUAL_ERROR,
                            "confidence": 0.6,
                            "details": f"비현실적 수치: {num_str}",
                            "method": "statistical"
                        })
            except ValueError:
                continue
        
        return detections
    
    async def _semantic_detection(self, content: str) -> List[Dict[str, Any]]:
        """의미적 환각 탐지"""
        detections = []
        
        # 의미적 모순 검사 (간단한 버전)
        contradictory_pairs = [
            (["증가", "늘어", "상승"], ["감소", "줄어", "하락"]),
            (["가능", "할 수"], ["불가능", "할 수 없"]),
            (["있다", "존재"], ["없다", "부재"]),
            (["항상", "언제나"], ["절대", "결코"])
        ]
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for positive_words, negative_words in contradictory_pairs:
                has_positive = any(word in sentence_lower for word in positive_words)
                has_negative = any(word in sentence_lower for word in negative_words)
                
                if has_positive and has_negative:
                    detections.append({
                        "type": HallucinationType.LOGICAL_CONTRADICTION,
                        "confidence": 0.7,
                        "text": sentence.strip(),
                        "details": "같은 문장 내 의미적 모순",
                        "method": "semantic"
                    })
        
        return detections
    
    def _get_number_context(self, content: str, number_str: str) -> str:
        """숫자 주변 문맥 추출"""
        index = content.find(number_str)
        if index == -1:
            return ""
        
        start = max(0, index - 50)
        end = min(len(content), index + len(number_str) + 50)
        
        return content[start:end]
    
    def _is_reasonable_large_number(self, context: str) -> bool:
        """큰 숫자가 합리적인지 판단"""
        reasonable_contexts = [
            "인구", "달러", "원", "개수", "거리", "면적", "데이터", "바이트"
        ]
        
        return any(ctx in context.lower() for ctx in reasonable_contexts)
    
    def _filter_by_confidence(self, detections: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """신뢰도 기반 필터링"""
        return [d for d in detections if d.get("confidence", 0.0) >= threshold]
    
    def _calculate_overall_confidence(self, detections: List[Dict[str, Any]]) -> float:
        """전체 신뢰도 계산"""
        if not detections:
            return 0.95  # 환각이 탐지되지 않으면 높은 신뢰도
        
        # 탐지된 환각의 평균 신뢰도
        avg_confidence = sum(d.get("confidence", 0.0) for d in detections) / len(detections)
        
        # 환각이 탐지될수록 전체 콘텐츠의 신뢰도는 낮아짐
        return max(0.1, 1.0 - avg_confidence)


class QualityAssessmentCapability(AgentCapability):
    """품질 평가 능력"""
    
    def get_capability_name(self) -> str:
        return "quality_assessment"
    
    def get_supported_formats(self) -> List[str]:
        return ["text", "structured_content", "reports", "responses"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        content = input_data
        assessment_criteria = config.get("criteria", list(EvaluationDimension))
        weights = config.get("weights", {})
        
        # 각 기준별 품질 평가
        quality_scores = await self._assess_quality_dimensions(content, assessment_criteria)
        
        # 가중 평균 계산
        weighted_score = self._calculate_weighted_score(quality_scores, weights)
        
        # 품질 등급 결정
        quality_grade = self._determine_quality_grade(weighted_score)
        
        return {
            "content_assessed": content,
            "dimension_scores": quality_scores,
            "weighted_overall_score": weighted_score,
            "quality_grade": quality_grade,
            "assessment_summary": self._generate_assessment_summary(quality_scores),
            "improvement_areas": self._identify_improvement_areas(quality_scores)
        }
    
    async def _assess_quality_dimensions(self, content: Any, criteria: List[EvaluationDimension]) -> Dict[str, float]:
        """품질 차원별 평가"""
        scores = {}
        content_str = str(content)
        
        for criterion in criteria:
            if criterion == EvaluationDimension.FACTUAL_ACCURACY:
                scores["factual_accuracy"] = await self._assess_factual_accuracy(content_str)
            elif criterion == EvaluationDimension.LOGICAL_CONSISTENCY:
                scores["logical_consistency"] = await self._assess_logical_consistency(content_str)
            elif criterion == EvaluationDimension.RELEVANCE:
                scores["relevance"] = await self._assess_relevance(content_str)
            elif criterion == EvaluationDimension.COMPLETENESS:
                scores["completeness"] = await self._assess_completeness(content_str)
            elif criterion == EvaluationDimension.CLARITY:
                scores["clarity"] = await self._assess_clarity(content_str)
            elif criterion == EvaluationDimension.OBJECTIVITY:
                scores["objectivity"] = await self._assess_objectivity(content_str)
        
        return scores
    
    async def _assess_factual_accuracy(self, content: str) -> float:
        """사실 정확성 평가"""
        # 간단한 휴리스틱 기반 평가
        accuracy_indicators = [
            (r'출처[:\s]*\[.*\]', 0.1),  # 출처 명시
            (r'연구.*따르면', 0.05),      # 연구 인용
            (r'\d{4}년', 0.03),          # 구체적 년도
            (r'약\s*\d+', 0.02),         # 근사값 표현
        ]
        
        base_score = 0.7
        for pattern, bonus in accuracy_indicators:
            if re.search(pattern, content):
                base_score += bonus
        
        # 의심스러운 패턴 페널티
        suspicious_patterns = [
            r'확실히.*아마도',
            r'모든.*하지만.*일부',
            r'항상.*때때로'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content):
                base_score -= 0.1
        
        return min(max(base_score, 0.0), 1.0)
    
    async def _assess_logical_consistency(self, content: str) -> float:
        """논리적 일관성 평가"""
        consistency_score = 0.8  # 기본 점수
        
        # 논리적 연결어 존재
        logical_connectors = ["따라서", "그러므로", "왜냐하면", "결과적으로", "반면"]
        connector_count = sum(1 for connector in logical_connectors if connector in content)
        consistency_score += min(connector_count * 0.02, 0.1)
        
        # 모순 패턴 감점
        contradictions = [
            (r'모든', r'일부'),
            (r'항상', r'때때로'),
            (r'절대.*않다', r'가능.*있다')
        ]
        
        for pos_pattern, neg_pattern in contradictions:
            if re.search(pos_pattern, content) and re.search(neg_pattern, content):
                consistency_score -= 0.15
        
        return min(max(consistency_score, 0.0), 1.0)
    
    async def _assess_relevance(self, content: str) -> float:
        """관련성 평가"""
        # 키워드 밀도 기반 간단 평가
        content_length = len(content.split())
        
        if content_length == 0:
            return 0.0
        
        # 기본 관련성 점수
        relevance_score = 0.75
        
        # 길이 기반 조정
        if content_length < 50:
            relevance_score -= 0.1  # 너무 짧으면 불완전할 가능성
        elif content_length > 1000:
            relevance_score -= 0.05  # 너무 길면 관련 없는 내용 포함 가능성
        
        return min(max(relevance_score, 0.0), 1.0)
    
    async def _assess_completeness(self, content: str) -> float:
        """완전성 평가"""
        completeness_indicators = [
            "결론",
            "요약",
            "정리하면",
            "마지막으로",
            "종합하면"
        ]
        
        structure_indicators = [
            "첫째",
            "둘째",
            "다음으로",
            "또한",
            "추가로"
        ]
        
        base_score = 0.6
        
        # 결론 존재성
        if any(indicator in content for indicator in completeness_indicators):
            base_score += 0.15
        
        # 구조적 완성도
        structure_count = sum(1 for indicator in structure_indicators if indicator in content)
        base_score += min(structure_count * 0.05, 0.2)
        
        # 길이 기반 완성도
        content_length = len(content.split())
        if content_length >= 100:
            base_score += 0.05
        
        return min(max(base_score, 0.0), 1.0)
    
    async def _assess_clarity(self, content: str) -> float:
        """명확성 평가"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # 평균 문장 길이
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        clarity_score = 0.7
        
        # 최적 문장 길이 (15-25 단어)
        if 15 <= avg_sentence_length <= 25:
            clarity_score += 0.15
        elif 10 <= avg_sentence_length <= 30:
            clarity_score += 0.1
        else:
            clarity_score -= 0.1
        
        # 복잡한 단어 비율
        all_words = content.split()
        complex_words = [word for word in all_words if len(word) > 8]
        complexity_ratio = len(complex_words) / max(len(all_words), 1)
        
        if complexity_ratio < 0.2:
            clarity_score += 0.1
        elif complexity_ratio > 0.4:
            clarity_score -= 0.15
        
        return min(max(clarity_score, 0.0), 1.0)
    
    async def _assess_objectivity(self, content: str) -> float:
        """객관성 평가"""
        subjective_indicators = [
            "생각합니다",
            "느낍니다",
            "개인적으로",
            "제 의견",
            "믿습니다",
            "추측"
        ]
        
        objective_indicators = [
            "연구에 따르면",
            "데이터에 의하면",
            "통계적으로",
            "조사 결과",
            "보고서에서"
        ]
        
        subjective_count = sum(1 for indicator in subjective_indicators if indicator in content)
        objective_count = sum(1 for indicator in objective_indicators if indicator in content)
        
        total_sentences = len([s for s in content.split('.') if s.strip()])
        
        objectivity_score = 0.8
        
        # 주관적 표현 페널티
        if total_sentences > 0:
            subjective_ratio = subjective_count / total_sentences
            objectivity_score -= subjective_ratio * 0.3
        
        # 객관적 표현 보너스
        if objective_count > 0:
            objectivity_score += min(objective_count * 0.05, 0.15)
        
        return min(max(objectivity_score, 0.0), 1.0)
    
    def _calculate_weighted_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """가중 점수 계산"""
        if not weights:
            # 동일 가중치
            return sum(scores.values()) / len(scores) if scores else 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in scores.items():
            weight = weights.get(dimension, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_grade(self, score: float) -> str:
        """품질 등급 결정"""
        if score >= 0.95:
            return "A+"
        elif score >= 0.9:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.8:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.7:
            return "B-"
        elif score >= 0.6:
            return "C"
        else:
            return "D"
    
    def _generate_assessment_summary(self, scores: Dict[str, float]) -> str:
        """평가 요약 생성"""
        best_aspect = max(scores, key=scores.get) if scores else "없음"
        worst_aspect = min(scores, key=scores.get) if scores else "없음"
        
        return f"최고 영역: {best_aspect} ({scores.get(best_aspect, 0):.2f}), 개선 필요 영역: {worst_aspect} ({scores.get(worst_aspect, 0):.2f})"
    
    def _identify_improvement_areas(self, scores: Dict[str, float]) -> List[str]:
        """개선 영역 식별"""
        improvement_areas = []
        
        for dimension, score in scores.items():
            if score < 0.7:
                improvement_areas.append(dimension)
        
        return improvement_areas


class GuardrailEnforcementCapability(AgentCapability):
    """가드레일 강제 능력"""
    
    def get_capability_name(self) -> str:
        return "guardrail_enforcement"
    
    def get_supported_formats(self) -> List[str]:
        return ["text", "content", "responses", "user_inputs"]
    
    async def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        content = str(input_data)
        guardrail_rules = config.get("rules", ["safety", "privacy", "bias", "harm"])
        enforcement_level = config.get("enforcement_level", "strict")
        
        # 각 가드레일 규칙 검사
        violations = await self._check_all_guardrails(content, guardrail_rules)
        
        # 강제 수준에 따른 처리
        enforcement_action = self._determine_enforcement_action(violations, enforcement_level)
        
        return {
            "content_checked": content,
            "guardrails_applied": guardrail_rules,
            "violations_found": violations,
            "enforcement_level": enforcement_level,
            "enforcement_action": enforcement_action,
            "safe_to_proceed": len(violations) == 0,
            "filtered_content": self._apply_content_filtering(content, violations) if violations else content
        }
    
    async def _check_all_guardrails(self, content: str, rules: List[str]) -> List[Dict[str, Any]]:
        """모든 가드레일 규칙 검사"""
        violations = []
        
        for rule in rules:
            if rule == "safety":
                safety_violations = await self._check_safety_guardrails(content)
                violations.extend(safety_violations)
            elif rule == "privacy":
                privacy_violations = await self._check_privacy_guardrails(content)
                violations.extend(privacy_violations)
            elif rule == "bias":
                bias_violations = await self._check_bias_guardrails(content)
                violations.extend(bias_violations)
            elif rule == "harm":
                harm_violations = await self._check_harm_guardrails(content)
                violations.extend(harm_violations)
        
        return violations
    
    async def _check_safety_guardrails(self, content: str) -> List[Dict[str, Any]]:
        """안전성 가드레일 검사"""
        violations = []
        
        safety_patterns = [
            {
                "pattern": r'(폭력|살해|공격|테러)',
                "severity": "high",
                "description": "폭력적 내용"
            },
            {
                "pattern": r'(자살|자해|상처)',
                "severity": "high", 
                "description": "자해 관련 내용"
            },
            {
                "pattern": r'(불법|범죄|위법)',
                "severity": "medium",
                "description": "불법 활동 언급"
            }
        ]
        
        for pattern_info in safety_patterns:
            if re.search(pattern_info["pattern"], content):
                violations.append({
                    "rule": "safety",
                    "type": pattern_info["description"],
                    "severity": pattern_info["severity"],
                    "pattern": pattern_info["pattern"],
                    "confidence": 0.8
                })
        
        return violations
    
    async def _check_privacy_guardrails(self, content: str) -> List[Dict[str, Any]]:
        """개인정보 보호 가드레일 검사"""
        violations = []
        
        privacy_patterns = [
            {
                "pattern": r'\d{3}-\d{4}-\d{4}',
                "type": "전화번호",
                "severity": "high"
            },
            {
                "pattern": r'\d{6}-[1-4]\d{6}',
                "type": "주민등록번호",
                "severity": "high"
            },
            {
                "pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "type": "이메일 주소",
                "severity": "medium"
            },
            {
                "pattern": r'신용카드.*\d{4}.*\d{4}.*\d{4}.*\d{4}',
                "type": "신용카드 번호",
                "severity": "high"
            }
        ]
        
        for pattern_info in privacy_patterns:
            matches = re.finditer(pattern_info["pattern"], content)
            for match in matches:
                violations.append({
                    "rule": "privacy",
                    "type": pattern_info["type"],
                    "severity": pattern_info["severity"],
                    "location": match.span(),
                    "matched_text": match.group(),
                    "confidence": 0.9
                })
        
        return violations
    
    async def _check_bias_guardrails(self, content: str) -> List[Dict[str, Any]]:
        """편향 가드레일 검사"""
        violations = []
        
        bias_indicators = [
            {
                "patterns": [r'모든\s+(남자|여자)', r'(남자|여자)들은\s+다'],
                "type": "성별 편향",
                "severity": "medium"
            },
            {
                "patterns": [r'(특정국가).*모두', r'(특정민족).*항상'],
                "type": "인종/민족 편향",
                "severity": "high"
            },
            {
                "patterns": [r'(젊은이|노인).*다\s+그렇다', r'세대.*특징.*모두'],
                "type": "연령 편향",
                "severity": "medium"
            }
        ]
        
        for bias_info in bias_indicators:
            for pattern in bias_info["patterns"]:
                if re.search(pattern, content):
                    violations.append({
                        "rule": "bias",
                        "type": bias_info["type"],
                        "severity": bias_info["severity"],
                        "pattern": pattern,
                        "confidence": 0.7
                    })
        
        return violations
    
    async def _check_harm_guardrails(self, content: str) -> List[Dict[str, Any]]:
        """유해성 가드레일 검사"""
        violations = []
        
        harm_categories = [
            {
                "keywords": ["혐오", "차별", "배제", "멸시"],
                "type": "혐오 표현",
                "severity": "high"
            },
            {
                "keywords": ["거짓말", "조작", "허위", "가짜"],
                "type": "허위정보",
                "severity": "medium"
            },
            {
                "keywords": ["위험한", "해로운", "독성", "중독"],
                "type": "위험 정보",
                "severity": "high"
            }
        ]
        
        for harm_info in harm_categories:
            for keyword in harm_info["keywords"]:
                if keyword in content:
                    violations.append({
                        "rule": "harm",
                        "type": harm_info["type"],
                        "severity": harm_info["severity"],
                        "keyword": keyword,
                        "confidence": 0.6
                    })
        
        return violations
    
    def _determine_enforcement_action(self, violations: List[Dict[str, Any]], level: str) -> str:
        """강제 조치 결정"""
        if not violations:
            return "allow"
        
        high_severity_count = sum(1 for v in violations if v.get("severity") == "high")
        medium_severity_count = sum(1 for v in violations if v.get("severity") == "medium")
        
        if level == "strict":
            if high_severity_count > 0:
                return "block"
            elif medium_severity_count > 0:
                return "filter"
            else:
                return "warn"
        elif level == "moderate":
            if high_severity_count >= 2:
                return "block"
            elif high_severity_count > 0 or medium_severity_count >= 3:
                return "filter"
            else:
                return "warn"
        else:  # lenient
            if high_severity_count >= 3:
                return "block"
            elif high_severity_count >= 2:
                return "filter"
            else:
                return "allow_with_warning"
    
    def _apply_content_filtering(self, content: str, violations: List[Dict[str, Any]]) -> str:
        """콘텐츠 필터링 적용"""
        filtered_content = content
        
        for violation in violations:
            if "matched_text" in violation:
                # 매칭된 텍스트를 마스킹
                matched_text = violation["matched_text"]
                mask = "*" * len(matched_text)
                filtered_content = filtered_content.replace(matched_text, mask)
            elif "keyword" in violation:
                # 키워드를 마스킹
                keyword = violation["keyword"]
                mask = "*" * len(keyword)
                filtered_content = filtered_content.replace(keyword, mask)
        
        return filtered_content


class EvaluatorAgent(ModularAgent):
    """LangChain 기반 응답 평가 에이전트"""
    
    def __init__(self, config: AgentConfig):
        # 프레임워크 검증
        if config.framework != AgentFramework.LANGCHAIN:
            raise ValueError(f"EvaluatorAgent는 LangChain 프레임워크만 지원합니다. 현재: {config.framework}")
        
        super().__init__(config)
        
        # 평가 설정
        self.quality_thresholds = config.custom_config.get("quality_thresholds", {
            'accuracy': 0.8,
            'completeness': 0.7,
            'relevance': 0.75,
            'safety': 0.9
        })
        self.guardrail_enforcement_level = config.custom_config.get("enforcement_level", "strict")
        
    async def _register_default_capabilities(self):
        """기본 능력 등록"""
        # 환각 탐지 능력
        hallucination_capability = HallucinationDetectionCapability()
        self.capability_registry.register_capability(hallucination_capability)
        
        # 품질 평가 능력
        quality_capability = QualityAssessmentCapability()
        self.capability_registry.register_capability(quality_capability)
        
        # 가드레일 강제 능력
        guardrail_capability = GuardrailEnforcementCapability()
        self.capability_registry.register_capability(guardrail_capability)
        
        self.logger.info("EvaluatorAgent 기본 능력 등록 완료")
    
    async def _load_framework_adapter(self) -> FrameworkAdapter:
        """LangChain 어댑터 로드"""
        adapter = LangChainAdapter()
        
        # 에이전트별 맞춤 설정
        framework_config = {
            "openai_api_key": self.config.custom_config.get("openai_api_key", "default-key"),
            "model_name": self.config.custom_config.get("model_name", "gpt-3.5-turbo"),
            "temperature": self.config.custom_config.get("temperature", 0.1),
            "max_tokens": self.config.custom_config.get("max_tokens", 1000),
            "verbose": self.config.custom_config.get("verbose", False)
        }
        
        await adapter.initialize(framework_config)
        return adapter
    
    async def comprehensive_evaluation(self, content: Any, evaluation_criteria: List[str]) -> Dict[str, Any]:
        """종합적 평가"""
        evaluation_request = ProcessingRequest(
            request_id=f"comprehensive_eval_{int(time.time())}",
            content=content,
            content_type="comprehensive_evaluation",
            processing_options={
                "evaluation_criteria": evaluation_criteria,
                "include_hallucination_check": True,
                "include_guardrail_check": True,
                "quality_thresholds": self.quality_thresholds
            }
        )
        
        response = await self.process(evaluation_request)
        
        return {
            "original_content": content,
            "evaluation_report": response.processed_content,
            "overall_confidence": response.confidence_score,
            "quality_metrics": response.quality_metrics,
            "evaluation_metadata": response.metadata,
            "recommendation": "approve" if response.confidence_score >= 0.8 else "revise"
        }
    
    async def detect_hallucinations_advanced(self, content: Any, reference_data: Optional[str] = None) -> Dict[str, Any]:
        """고급 환각 탐지"""
        hallucination_capability = self.capability_registry.get_capability("hallucination_detection")
        
        if hallucination_capability:
            result = await hallucination_capability.execute(content, {
                "detection_method": "comprehensive",
                "confidence_threshold": 0.7,
                "reference_data": reference_data
            })
            
            return result
        else:
            return {
                "error": "환각 탐지 능력을 사용할 수 없습니다",
                "hallucinations_detected": False
            }
    
    async def enforce_safety_guardrails(self, content: Any, rules: List[str]) -> Dict[str, Any]:
        """안전 가드레일 강제"""
        guardrail_capability = self.capability_registry.get_capability("guardrail_enforcement")
        
        if guardrail_capability:
            result = await guardrail_capability.execute(content, {
                "rules": rules,
                "enforcement_level": self.guardrail_enforcement_level
            })
            
            return result
        else:
            return {
                "error": "가드레일 강제 능력을 사용할 수 없습니다",
                "safe_to_proceed": True
            }


# 사용 예시
async def example_evaluator_usage():
    """EvaluatorAgent 사용 예시"""
    
    # 설정 생성
    config = AgentConfig(
        agent_id="evaluator_main",
        framework=AgentFramework.LANGCHAIN,
        capabilities=["hallucination_detection", "quality_assessment", "guardrail_enforcement"],
        processing_mode=ProcessingMode.ASYNC,
        custom_config={
            "quality_thresholds": {
                "accuracy": 0.85,
                "completeness": 0.8,
                "relevance": 0.8,
                "safety": 0.95
            },
            "enforcement_level": "strict",
            "openai_api_key": "your-api-key",
            "model_name": "gpt-3.5-turbo"
        }
    )
    
    # 에이전트 생성
    evaluator = EvaluatorAgent(config)
    
    # 평가할 샘플 콘텐츠
    sample_content = """
    Python은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어입니다.
    간단하고 읽기 쉬운 구문을 가지고 있어 초보자에게 인기가 높습니다.
    웹 개발, 데이터 분석, 인공지능 분야에서 널리 사용됩니다.
    Django와 Flask는 대표적인 Python 웹 프레임워크입니다.
    """
    
    # 기본 평가 요청
    request = ProcessingRequest(
        request_id="test_evaluation_001",
        content=sample_content,
        content_type="content_evaluation",
        processing_options={
            "evaluation_criteria": ["factual_accuracy", "logical_consistency", "completeness"],
            "original_query": "Python 프로그래밍 언어에 대해 설명해주세요"
        }
    )
    
    # 평가 실행
    response = await evaluator.process(request)
    
    print(f"평가 완료: 신뢰도 {response.confidence_score:.2f}")
    print(f"사용된 프레임워크: {response.framework_info['name']}")
    print(f"처리 시간: {response.processing_time:.2f}초")
    
    # 종합적 평가 테스트
    comprehensive_result = await evaluator.comprehensive_evaluation(
        sample_content,
        ["factual_accuracy", "logical_consistency", "relevance", "safety"]
    )
    
    print(f"종합 평가 완료: {comprehensive_result['recommendation']}")
    print(f"전체 신뢰도: {comprehensive_result['overall_confidence']:.2f}")
    
    # 환각 탐지 테스트
    hallucination_result = await evaluator.detect_hallucinations_advanced(
        "Python은 2025년에 새로 개발된 언어입니다."  # 의도적 잘못된 정보
    )
    
    print(f"환각 탐지: {hallucination_result.get('hallucinations_detected', False)}")
    
    # 가드레일 테스트
    safety_result = await evaluator.enforce_safety_guardrails(
        sample_content,
        ["safety", "privacy", "bias"]
    )
    
    print(f"안전성 검사: {safety_result.get('safe_to_proceed', True)}")
    
    return response


if __name__ == "__main__":
    asyncio.run(example_evaluator_usage())