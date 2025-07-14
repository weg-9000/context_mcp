import logging
import time
from typing import List, Dict, Any
import sys
import io
import os
from .agent_decision_logger import AgentDecisionLogger
from datetime import datetime

if sys.platform.startswith('win'):
    # Windows에서 UTF-8 인코딩 강제 설정
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

class HybridLogger:
    """범용 하이브리드 로깅 시스템 - 설정 기반 초기화"""
    
    def __init__(self, name: str = None, config: Dict = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(self.name)
        self.agent_logger = None
        
        # 설정 기반 초기화
        if self.config.get('disable_file_logging', os.getenv('DISABLE_FILE_LOGGING', 'false').lower() == 'true'):
            self.disable_file_logging()
            
        self.setup_logging()

    def disable_file_logging(self):
        """파일 로깅 완전 비활성화"""
        root_logger = logging.getLogger()
        # 모든 파일 핸들러 제거
        for handler in root_logger.handlers[:]:
            if isinstance(handler, (logging.FileHandler, logging.handlers.TimedRotatingFileHandler)):
                root_logger.removeHandler(handler)

        # 새로운 파일 핸들러 추가 방지
        original_addHandler = root_logger.addHandler

        def filtered_addHandler(handler):
            if not isinstance(handler, (logging.FileHandler, logging.handlers.TimedRotatingFileHandler)):
                original_addHandler(handler)

        root_logger.addHandler = filtered_addHandler

    def safe_log(self, message):
        """안전한 로깅"""
        try:
            if hasattr(self, 'logger'):
                self.logger.info(message)
        except (ValueError, OSError):
            # 파일 로깅 비활성화 - 콘솔 출력만 시도
            try:
                print(f"{datetime.now()}: {message}")
            except:
                pass  # 완전 실패 시 무시

    def setup_logging(self):
        """로깅 설정 (설정 기반)"""
        try:
            # 설정에서 에이전트 로거 구성 가져오기
            agent_config = self.config.get('agent_logger', {})
            self.agent_logger = AgentDecisionLogger(agent_config)
        except ImportError:
            self.logger.warning("AgentDecisionLogger를 불러올 수 없습니다. 표준 로깅으로 진행합니다.")

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)

    def log_agent_decision(self, agent_name: str, agent_role: str = None,
                          input_data: Dict = None, decision_process: Dict = None,
                          output_result: Dict = None, reasoning: str = "",
                          confidence_score: float = 0.8, context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """에이전트 결정 로깅 - AgentDecisionLogger로 위임"""
        if self.agent_logger:
            return self.agent_logger.log_agent_decision(
                agent_name=agent_name,
                agent_role=agent_role,
                input_data=input_data,
                decision_process=decision_process,
                output_result=output_result,
                reasoning=reasoning,
                confidence_score=confidence_score,
                context=context,
                performance_metrics=performance_metrics
            )
        else:
            self.info(f"Agent Decision - {agent_name}: {reasoning}")
            return f"standard_{agent_name}_{int(time.time())}"

    def log_agent_real_output(self, agent_name: str, agent_role: str = None,
                             task_description: str = "", final_answer: str = "",
                             reasoning_process: str = "", execution_steps: List[str] = None,
                             raw_input: Any = None, raw_output: Any = None,
                             performance_metrics: Dict = None, error_logs: List[Dict] = None,
                             info_data: Dict = None) -> str:
        """에이전트 실제 출력 로깅 - AgentDecisionLogger로 위임"""
        if self.agent_logger:
            return self.agent_logger.log_agent_real_output(
                agent_name=agent_name,
                agent_role=agent_role,
                task_description=task_description,
                final_answer=final_answer,
                reasoning_process=reasoning_process,
                execution_steps=execution_steps,
                raw_input=raw_input,
                raw_output=raw_output,
                performance_metrics=performance_metrics,
                error_logs=error_logs,
                info_data=info_data
            )
        else:
            self.info(f"Agent Output - {agent_name}: {task_description}")
            return f"standard_{agent_name}_{int(time.time())}"

    def log_agent_interaction(self, source_agent: str, target_agent: str,
                             interaction_type: str, data_transferred: Dict,
                             success: bool = True) -> str:
        """에이전트 간 상호작용 로깅 - AgentDecisionLogger로 위임"""
        if self.agent_logger:
            return self.agent_logger.log_agent_interaction(
                source_agent=source_agent,
                target_agent=target_agent,
                interaction_type=interaction_type,
                data_transferred=data_transferred,
                success=success
            )
        else:
            self.info(f"Agent Interaction - {source_agent} -> {target_agent}: {interaction_type}")
            return f"interaction_{int(time.time())}"

    # ==================== 범용 처리 메서드들 (설정 기반) ====================

    def log_processing_start(self, process_type: str, metadata: Dict = None):
        """범용 처리 시작 로깅"""
        try:
            log_data = {
                "event": f"{process_type}_processing_start",
                "process_type": process_type,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }

            self.info(f"{process_type} 처리 시작")
            
            # 에이전트 로거에도 기록
            return self.log_agent_real_output(
                agent_name=f"{process_type}System",
                agent_role=f"{process_type} 처리 시스템",
                task_description=f"{process_type} 처리 시작",
                final_answer="처리 시작됨",
                raw_input={"process_type": process_type},
                raw_output=log_data,
                performance_metrics={"processing_started": True}
            )

        except Exception as e:
            self.error(f"{process_type} 처리 시작 로깅 실패: {e}")
            return None

    def log_quality_report(self, quality_report: Dict, system_name: str = "QualitySystem"):
        """범용 품질 보고서 로깅"""
        try:
            overall_score = quality_report.get("overall_score", 0)
            self.info(f"품질 보고서 생성 완료 - 전체 점수: {overall_score:.2f}")

            # 에이전트 로거에 상세 기록
            return self.log_agent_real_output(
                agent_name=system_name,
                agent_role="품질 평가 시스템",
                task_description="품질 보고서 생성",
                final_answer=f"품질 평가 완료 (점수: {overall_score:.2f})",
                raw_input={"report_request": True},
                raw_output=quality_report,
                performance_metrics={
                    "overall_score": overall_score,
                    "data_integrity": quality_report.get("data_integrity", {}),
                    "report_generated": True
                }
            )
        except Exception as e:
            self.error(f"품질 보고서 로깅 실패: {e}")
            return None

    def log_data_integrity_check(self, check_result: Dict, checker_name: str = "IntegrityChecker"):
        """범용 데이터 무결성 검사 로깅"""
        try:
            integrity_score = check_result.get("overall_integrity_score", 0)
            self.info(f"데이터 무결성 검사 완료 - 무결성 점수: {integrity_score:.2f}")

            return self.log_agent_real_output(
                agent_name=checker_name,
                agent_role="데이터 무결성 검사기",
                task_description="데이터 무결성 검증",
                final_answer=f"무결성 검사 완료 (점수: {integrity_score:.2f})",
                raw_output=check_result,
                performance_metrics={
                    "integrity_score": integrity_score,
                    "check_passed": integrity_score > 0.9
                }
            )
        except Exception as e:
            self.error(f"데이터 무결성 검사 로깅 실패: {e}")
            return None

    def log_guideline_application(self, guidelines_data: Dict, applicator_name: str = "GuidelineApplicator"):
        """범용 가이드라인 적용 로깅"""
        try:
            confidence_score = guidelines_data.get("quality_metrics", {}).get("confidence_level", 0)
            self.info(f"가이드라인 적용 완료 - 신뢰도: {confidence_score:.2f}")

            return self.log_agent_real_output(
                agent_name=applicator_name,
                agent_role="가이드라인 적용기",
                task_description="가이드라인 적용",
                final_answer=f"가이드라인 적용 완료 (신뢰도: {confidence_score:.2f})",
                raw_output=guidelines_data,
                performance_metrics={
                    "confidence_score": confidence_score,
                    "guidelines_applied": True
                }
            )
        except Exception as e:
            self.error(f"가이드라인 적용 로깅 실패: {e}")
            return None

    # ==================== 유틸리티 메서드들 ====================

    def get_logger_status(self) -> Dict:
        """로거 상태 정보 반환"""
        return {
            "class_name": self.name,
            "standard_logger_name": self.logger.name,
            "agent_logger_available": self.agent_logger is not None,
            "agent_logger_type": type(self.agent_logger).__name__ if self.agent_logger else None,
            "handlers_count": len(self.logger.handlers),
            "log_level": self.logger.level,
            "config": self.config
        }

    def enable_hybrid_mode(self):
        """하이브리드 모드 활성화"""
        self.logger.info("하이브리드 로깅 모드 활성화")

    def disable_hybrid_mode(self):
        """하이브리드 모드 비활성화 (표준 로깅만 사용)"""
        self.logger.info("하이브리드 로깅 모드 비활성화 - 표준 로깅만 사용")

    def test_logging_system(self):
        """로깅 시스템 테스트"""
        try:
            self.info("=== 하이브리드 로깅 시스템 테스트 시작 ===")
            
            # 표준 로깅 테스트
            self.debug("DEBUG 레벨 테스트")
            self.info("INFO 레벨 테스트")
            self.warning("WARNING 레벨 테스트")
            self.error("ERROR 레벨 테스트")

            # 에이전트 로깅 테스트
            test_output_id = self.log_agent_real_output(
                agent_name="TestAgent",
                agent_role="테스트 에이전트",
                task_description="로깅 시스템 테스트",
                final_answer="테스트 성공",
                performance_metrics={"test_passed": True}
            )

            # 에이전트 결정 로깅 테스트
            test_decision_id = self.log_agent_decision(
                agent_name="TestDecisionAgent",
                agent_role="결정 테스트 에이전트",
                input_data={"test": True},
                decision_process={"steps": ["테스트 단계"]},
                output_result={"result": "성공"},
                reasoning="테스트 목적"
            )

            # 상태 정보 출력
            status = self.get_logger_status()
            self.info(f"로거 상태: {status}")
            self.info("=== 하이브리드 로깅 시스템 테스트 완료 ===")

            return {
                "test_passed": True,
                "output_id": test_output_id,
                "decision_id": test_decision_id,
                "status": status
            }

        except Exception as e:
            self.error(f"로깅 시스템 테스트 실패: {e}")
            return {"test_passed": False, "error": str(e)}

    def get_all_previous_results(self, agent_name: str = None) -> List[Dict]:
        """모든 이전 결과 조회 (개선됨)"""
        try:
            # 에이전트 로거가 있는 경우
            if self.agent_logger and hasattr(self.agent_logger, 'get_all_previous_results'):
                try:
                    if agent_name:
                        return self.agent_logger.get_all_previous_results(agent_name)
                    else:
                        all_outputs = self.agent_logger.output_manager.get_all_outputs()
                        if isinstance(all_outputs, list):
                            return all_outputs
                        
                except Exception as e:
                    self.warning(f"에이전트 로거에서 결과 조회 실패: {e}")

            # SafeAgentLogger인 경우
            if hasattr(self.agent_logger, 'outputs'):
                outputs = self.agent_logger.outputs
                if isinstance(outputs, list):
                    if agent_name:
                        filtered_outputs = []
                        for output in outputs:
                            if isinstance(output, dict) and agent_name.lower() in output.get('agent_name', '').lower():
                                filtered_outputs.append(output)
                        return filtered_outputs
                    return outputs

            # 최종 폴백: 빈 리스트
            self.info(f"이전 결과를 찾을 수 없음 (에이전트: {agent_name})")
            return []

        except Exception as e:
            self.error(f"이전 결과 조회 실패: {e}")
            return []

    def get_recent_outputs(self, agent_name: str = None, limit: int = 10) -> List[Dict]:
        """최근 출력 조회"""
        try:
            all_results = self.get_all_previous_results(agent_name)
            # 타임스탬프 기준으로 정렬
            sorted_results = sorted(
                all_results,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            return sorted_results[:limit]
        except Exception as e:
            self.error(f"최근 출력 조회 실패: {e}")
            return []

    def get_agent_statistics(self, agent_name: str = None) -> Dict:
        """에이전트 통계 조회 (개선됨)"""
        try:
            all_results = self.get_all_previous_results(agent_name)
            stats = {
                "total_outputs": len(all_results),
                "agents": {},
                "recent_activity": None,
                "error_count": 0,
                "performance_summary": {
                    "avg_confidence": 0,
                    "avg_response_length": 0
                }
            }

            # 에이전트별 통계
            confidence_scores = []
            response_lengths = []
            
            for result in all_results:
                if isinstance(result, dict):
                    agent = result.get('agent_name', 'unknown')
                    if agent not in stats["agents"]:
                        stats["agents"][agent] = 0
                    stats["agents"][agent] += 1

                    # 에러 카운트
                    if 'error' in result or result.get('final_answer', '').lower().find('error') != -1:
                        stats["error_count"] += 1
                        
                    # 성능 메트릭 수집
                    metrics = result.get('performance_metrics', {})
                    if 'confidence_score' in metrics:
                        confidence_scores.append(metrics['confidence_score'])
                    if 'response_length' in metrics:
                        response_lengths.append(metrics['response_length'])

                    # 최근 활동
                    timestamp = result.get('timestamp')
                    if timestamp and (not stats["recent_activity"] or timestamp > stats["recent_activity"]):
                        stats["recent_activity"] = timestamp

            # 성능 요약 계산
            if confidence_scores:
                stats["performance_summary"]["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
            if response_lengths:
                stats["performance_summary"]["avg_response_length"] = sum(response_lengths) / len(response_lengths)

            return stats

        except Exception as e:
            self.error(f"에이전트 통계 조회 실패: {e}")
            return {"total_outputs": 0, "agents": {}, "error_count": 0}

    def get_learning_insights(self, target_agent: str = None) -> Dict:
        """학습 인사이트 조회"""
        if self.agent_logger and hasattr(self.agent_logger, 'get_learning_insights'):
            return self.agent_logger.get_learning_insights(target_agent or "system")
        return {
            "target_agent": target_agent or "system",
            "analysis_timestamp": time.time(),
            "insights": "에이전트 로거를 사용할 수 없습니다.",
            "patterns": [],
            "recommendations": []
        }

class SafeAgentLogger:
    """안전한 에이전트 로거 (AgentLogger 없을 때 사용) - 개선됨"""

    def __init__(self, standard_logger, config: Dict = None):
        self.standard_logger = standard_logger
        self.config = config or {}
        self.outputs = []
        self.max_outputs = self.config.get('max_outputs', 1000)  # 메모리 사용량 제한

    def get_all_outputs(self) -> List[Dict]:
        """모든 출력 반환"""
        return self.outputs.copy()

    def get_outputs_by_agent(self, agent_name: str) -> List[Dict]:
        """특정 에이전트의 출력만 반환"""
        filtered_outputs = []
        for output in self.outputs:
            if isinstance(output, dict) and agent_name.lower() in output.get('agent_name', '').lower():
                filtered_outputs.append(output)
        return filtered_outputs

    def log_agent_decision(self, agent_name: str, agent_role: str = None,
                          input_data: Dict = None, decision_process: Dict = None,
                          output_result: Dict = None, reasoning: str = "",
                          confidence_score: float = 0.8, context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """안전한 폴백 에이전트 결정 로깅 (개선됨)"""
        try:
            output_id = f"safe_{agent_name}_{int(__import__('time').time() * 1000)}"
            log_entry = {
                "output_id": output_id,
                "agent_name": agent_name,
                "agent_role": agent_role or f"{agent_name} 에이전트",
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "input_data": input_data or {},
                "decision_process": decision_process or {},
                "output_result": output_result or {},
                "reasoning": reasoning,
                "confidence_score": confidence_score,
                "context": context or {},
                "performance_metrics": performance_metrics or {},
                "safe_mode": True
            }

            # 메모리 관리
            if len(self.outputs) >= self.max_outputs:
                self.outputs = self.outputs[-int(self.max_outputs * 0.8):]  # 20% 정리

            self.outputs.append(log_entry)
            self.standard_logger.info(f"Agent Decision (Safe Mode) - {agent_name}: {reasoning}")
            return output_id

        except Exception as e:
            self.standard_logger.error(f"안전 모드 에이전트 결정 로깅 실패: {e}")
            return f"safe_error_{agent_name}_{int(__import__('time').time())}"

    def log_agent_real_output(self, agent_name: str, agent_role: str = None,
                             task_description: str = "", final_answer: str = "",
                             reasoning_process: str = "", execution_steps: List[str] = None,
                             raw_input: Any = None, raw_output: Any = None,
                             performance_metrics: Dict = None, error_logs: List[Dict] = None,
                             info_data: Dict = None) -> str:
        """안전한 폴백 에이전트 출력 로깅 (개선됨)"""
        try:
            output_id = f"safe_output_{agent_name}_{int(__import__('time').time() * 1000)}"
            log_entry = {
                "output_id": output_id,
                "agent_name": agent_name,
                "agent_role": agent_role or f"{agent_name} 에이전트",
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "task_description": task_description,
                "final_answer": final_answer,
                "reasoning_process": reasoning_process,
                "execution_steps": execution_steps or [],
                "raw_input": raw_input,
                "raw_output": raw_output,
                "performance_metrics": performance_metrics or {},
                "error_logs": error_logs or [],
                "info_data": info_data or {},
                "safe_mode": True
            }

            # 메모리 관리
            if len(self.outputs) >= self.max_outputs:
                self.outputs = self.outputs[-int(self.max_outputs * 0.8):]

            self.outputs.append(log_entry)
            self.standard_logger.info(f"Agent Output (Safe Mode) - {agent_name}: {task_description}")
            return output_id

        except Exception as e:
            self.standard_logger.error(f"안전 모드 에이전트 출력 로깅 실패: {e}")
            return f"safe_output_error_{agent_name}_{int(__import__('time').time())}"

    def log_agent_interaction(self, source_agent: str, target_agent: str,
                             interaction_type: str, data_transferred: Dict,
                             success: bool = True) -> str:
        """안전한 폴백 에이전트 상호작용 로깅"""
        try:
            interaction_id = f"safe_interaction_{int(time.time() * 1000)}"
            self.standard_logger.info(
                f"Agent Interaction (Safe Mode) - {source_agent} -> {target_agent}: {interaction_type}"
            )
            return interaction_id
        except Exception as e:
            self.standard_logger.error(f"안전 모드 상호작용 로깅 실패: {e}")
            return f"safe_interaction_error_{int(time.time())}"

    def get_learning_insights(self, target_agent: str = None) -> Dict:
        """안전한 폴백 인사이트"""
        return {
            "target_agent": target_agent or "unknown",
            "analysis_timestamp": time.time(),
            "total_outputs_analyzed": len(self.outputs),
            "patterns": ["safe_mode"],
            "recommendations": ["안전 모드에서 실행 중"],
            "key_insights": "SafeAgentLogger 모드",
            "safe_mode": True
        }

# ==================== 팩토리 함수들 (설정 기반으로 개선) ====================

def create_hybrid_logger(class_name: str, config: Dict = None, agent_logger_factory=None) -> HybridLogger:
    """하이브리드 로거 생성 팩토리 함수 (설정 지원)"""
    try:
        # 에이전트 로거 생성
        if agent_logger_factory:
            agent_logger = agent_logger_factory()
        else:
            try:
                from .agent_decision_logger import get_agent_logger
                agent_config = config.get('agent_logger', {}) if config else {}
                agent_logger = get_agent_logger(agent_config)
            except ImportError:
                agent_logger = None

        # 하이브리드 로거 생성
        hybrid_logger = HybridLogger(class_name, config)
        hybrid_logger.agent_logger = agent_logger

        return hybrid_logger

    except Exception as e:
        # 최종 폴백: 표준 로거만 사용
        print(f"하이브리드 로거 생성 실패, 표준 로거 사용: {e}")
        return HybridLogger(class_name, config)

def get_hybrid_logger(class_name: str = None, config: Dict = None) -> HybridLogger:
    """하이브리드 로거 싱글톤 인스턴스 반환 (설정 지원)"""
    global _hybrid_logger_instances
    if '_hybrid_logger_instances' not in globals():
        _hybrid_logger_instances = {}

    # 설정을 포함한 키 생성
    config_key = f"{class_name}_{hash(str(sorted(config.items()))) if config else 'default'}"
    
    if config_key not in _hybrid_logger_instances:
        _hybrid_logger_instances[config_key] = create_hybrid_logger(
            class_name or "DefaultHybridLogger", 
            config
        )
    
    return _hybrid_logger_instances[config_key]

def setup_hybrid_logging_for_class(cls, config: Dict = None):
    """클래스 데코레이터: 하이브리드 로깅 자동 설정 (설정 지원)"""
    class_name = cls.__name__

    # 클래스에 logger 속성 추가
    cls.logger = get_hybrid_logger(class_name, config)

    # 기존 메서드들에 로깅 래퍼 추가 (선택적)
    original_init = cls.__init__

    def wrapped_init(self, *args, **kwargs):
        self.logger = get_hybrid_logger(class_name, config)
        self.logger.info(f"{class_name} 인스턴스 초기화")
        return original_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init
    return cls

# ==================== 전역 인스턴스 관리 ====================

_hybrid_logger_instances = {}

def reset_hybrid_loggers():
    """모든 하이브리드 로거 인스턴스 리셋"""
    global _hybrid_logger_instances
    _hybrid_logger_instances = {}

def get_all_hybrid_loggers() -> Dict[str, HybridLogger]:
    """모든 하이브리드 로거 인스턴스 반환"""
    return _hybrid_logger_instances.copy()

# ==================== 테스트 함수 ====================

def test_hybrid_logging_system(config: Dict = None):
    """하이브리드 로깅 시스템 전체 테스트 (설정 지원)"""
    print("=== 하이브리드 로깅 시스템 전체 테스트 시작 ===")
    
    try:
        # 하이브리드 로거 생성 테스트
        logger = get_hybrid_logger("TestSystem", config)

        # 로깅 시스템 테스트
        test_result = logger.test_logging_system()
        print(f"테스트 결과: {test_result}")
        print("=== 하이브리드 로깅 시스템 전체 테스트 완료 ===")

        return test_result

    except Exception as e:
        print(f"하이브리드 로깅 시스템 테스트 실패: {e}")
        return {"test_passed": False, "error": str(e)}

if __name__ == "__main__":
    # 테스트 실행
    test_config = {
        'agent_logger': {
            'database_type': 'filesystem',
            'filesystem_path': './test_logs',
            'max_memory_cache': 50
        },
        'disable_file_logging': False
    }
    test_hybrid_logging_system(test_config)
