import asyncio
import json
import time
from typing import Dict, Any, TYPE_CHECKING, List, Optional, Callable
from abc import ABC, abstractmethod

from .hybridlogging import HybridLogger

class AgentResponseProcessor(ABC):
    
    @abstractmethod
    def process_response(self, response_data: Any) -> str:
        pass

class DefaultResponseProcessor(AgentResponseProcessor):
    
    def process_response(self, response_data: Any) -> str:
        try:
            if isinstance(response_data, str):
                return response_data
            elif isinstance(response_data, dict):
                return json.dumps(response_data, ensure_ascii=False, indent=2)
            elif isinstance(response_data, list):
                return json.dumps(response_data, ensure_ascii=False, indent=2)
            else:
                return str(response_data)
        except Exception as e:
            return f"응답 데이터 처리 실패: {str(e)}"

class JsonResponseProcessor(AgentResponseProcessor):
    
    def process_response(self, response_data: Any) -> str:
        try:
            return json.dumps(response_data, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"JSON 처리 실패: {str(e)}"

class StructuredResponseProcessor(AgentResponseProcessor):
    
    def process_response(self, response_data: Any) -> str:
        try:
            if isinstance(response_data, dict):
                if "answer" in response_data:
                    return f"응답: {response_data['answer']}"
                elif "result" in response_data:
                    return f"결과: {response_data['result']}"
                elif "content" in response_data:
                    return f"내용: {response_data['content']}"
                else:
                    return json.dumps(response_data, ensure_ascii=False, indent=2)
            return str(response_data)
        except Exception as e:
            return f"구조화 처리 실패: {str(e)}"

class LoggingManager:
    
    def __init__(self, logger: 'HybridLogger', config: Dict = None):
        self.logger = logger
        self.config = config or {}
        self.response_counter = 0
        
        self.response_processor = self._create_response_processor()
        self.agent_configs = self.config.get('agent_configs', {})
        
        self.pre_log_callbacks: List[Callable] = []
        self.post_log_callbacks: List[Callable] = []
        
        self.performance_tracker = {}
        self.error_threshold = self.config.get('error_threshold', 0.1)
        self.max_response_length = self.config.get('max_response_length', 10000)

    def _create_response_processor(self) -> AgentResponseProcessor:
        processor_type = self.config.get('response_processor', 'default')
        
        if processor_type == 'json':
            return JsonResponseProcessor()
        elif processor_type == 'structured':
            return StructuredResponseProcessor()
        else:
            return DefaultResponseProcessor()

    def register_pre_log_callback(self, callback: Callable):
        self.pre_log_callbacks.append(callback)

    def register_post_log_callback(self, callback: Callable):
        self.post_log_callbacks.append(callback)

    def _execute_callbacks(self, callbacks: List[Callable], *args, **kwargs):
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"콜백 실행 실패: {e}")

    def _track_performance(self, agent_name: str, metrics: Dict):
        if agent_name not in self.performance_tracker:
            self.performance_tracker[agent_name] = {
                "total_requests": 0,
                "total_errors": 0,
                "avg_response_time": 0,
                "response_times": []
            }
        
        tracker = self.performance_tracker[agent_name]
        tracker["total_requests"] += 1
        
        if metrics.get("error", False):
            tracker["total_errors"] += 1
        
        if "processing_time" in metrics:
            response_time = metrics["processing_time"]
            tracker["response_times"].append(response_time)
            
            if len(tracker["response_times"]) > 100:
                tracker["response_times"] = tracker["response_times"][-50:]
            
            tracker["avg_response_time"] = sum(tracker["response_times"]) / len(tracker["response_times"])

    def _validate_response_data(self, response_data: Any) -> bool:
        try:
            response_str = str(response_data)
            if len(response_str) > self.max_response_length:
                self.logger.warning(f"응답 데이터가 너무 큽니다: {len(response_str)} 문자")
                return False
            return True
        except Exception:
            return False

    async def log_agent_response(self, agent_name: str, agent_role: str,
                                task_description: str, response_data: Any,
                                metadata: Dict = None) -> str:
        try:
            start_time = time.time()
            response_id = f"response_{self.response_counter}_{int(time.time())}"
            
            if not self._validate_response_data(response_data):
                response_data = {"error": "응답 데이터가 너무 크거나 유효하지 않습니다"}
            
            self._execute_callbacks(
                self.pre_log_callbacks,
                agent_name, response_data, metadata
            )

            processed_response = self.response_processor.process_response(response_data)
            
            agent_config = self.agent_configs.get(agent_name, {})
            
            performance_metrics = {
                "response_length": len(str(processed_response)),
                "response_type": type(response_data).__name__,
                "log_entry_id": response_id,
                "processing_time": time.time() - start_time,
                "agent_config_applied": bool(agent_config)
            }
            
            if agent_config.get('track_confidence', False) and isinstance(response_data, dict):
                performance_metrics["confidence_score"] = response_data.get('confidence', 0.8)
                
            if agent_config.get('track_tokens', False) and isinstance(response_data, dict):
                performance_metrics["token_count"] = response_data.get('token_count', 0)
                
            if agent_config.get('track_errors', True) and isinstance(response_data, dict):
                if "error" in response_data or "exception" in response_data:
                    performance_metrics["error"] = True

            self._track_performance(agent_name, performance_metrics)

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.logger.log_agent_real_output(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    task_description=task_description,
                    final_answer=processed_response,
                    performance_metrics=performance_metrics,
                    info_data=metadata or {}
                )
            )
            
            self._execute_callbacks(
                self.post_log_callbacks,
                agent_name, response_id, processed_response
            )

            self.response_counter += 1
            return response_id

        except Exception as e:
            self.logger.error(f"에이전트 응답 로그 저장 실패 {agent_name}: {e}")
            return "log_save_failed"

    async def log_processing_completion(self, agent_name: str, agent_role: str, 
                                       process_type: str, result_data: Dict,
                                       metadata: Dict = None):
        task_description = f"{process_type} 처리 완료"
        
        metrics = {
            "process_type": process_type,
            "success": result_data.get("success", True),
            "items_processed": result_data.get("items_count", 0),
            "processing_time": result_data.get("processing_time", 0),
            "completion_status": "success" if result_data.get("success", True) else "failed"
        }
        
        await self.log_agent_response(
            agent_name=agent_name,
            agent_role=agent_role,
            task_description=task_description,
            response_data=result_data,
            metadata={**(metadata or {}), "metrics": metrics}
        )

    async def log_analysis_completion(self, agent_name: str, analysis_result: Dict, 
                                     analysis_type: str = "analysis"):
        await self.log_processing_completion(
            agent_name=agent_name,
            agent_role=f"{analysis_type.title()} 분석기",
            process_type=f"{analysis_type}_analysis",
            result_data=analysis_result,
            metadata={"stage": f"{analysis_type}_analysis", "async_processing": True}
        )

    async def log_generation_completion(self, agent_name: str, generation_result: Dict,
                                       generation_type: str = "content"):
        await self.log_processing_completion(
            agent_name=agent_name,
            agent_role=f"{generation_type.title()} 생성기",
            process_type=f"{generation_type}_generation",
            result_data=generation_result,
            metadata={"stage": f"{generation_type}_generation", "async_processing": True}
        )

    async def log_validation_completion(self, agent_name: str, validation_result: Dict,
                                       validation_type: str = "validation"):
        await self.log_processing_completion(
            agent_name=agent_name,
            agent_role=f"{validation_type.title()} 검증기",
            process_type=f"{validation_type}_validation",
            result_data=validation_result,
            metadata={"stage": f"{validation_type}_validation", "async_processing": True}
        )

    async def log_transformation_completion(self, agent_name: str, transform_result: Dict,
                                           transform_type: str = "transform"):
        await self.log_processing_completion(
            agent_name=agent_name,
            agent_role=f"{transform_type.title()} 변환기",
            process_type=f"{transform_type}_transformation",
            result_data=transform_result,
            metadata={"stage": f"{transform_type}_transformation", "async_processing": True}
        )

    async def log_image_analysis_completion(self, images_count: int, results_count: int):
        await self.log_analysis_completion(
            agent_name="ImageAnalyzerAgent",
            analysis_result={
                "images_processed": images_count,
                "results_generated": results_count,
                "success_rate": results_count / images_count if images_count > 0 else 0,
                "processing_efficiency": "high" if results_count / images_count > 0.9 else "medium"
            },
            analysis_type="image"
        )

    async def log_content_creation_completion(self, texts_count: int, images_count: int, content_length: int):
        await self.log_generation_completion(
            agent_name="ContentCreatorV2Agent",
            generation_result={
                "texts_count": texts_count,
                "images_count": images_count,
                "content_length": content_length,
                "content_efficiency": content_length / max(texts_count + images_count, 1)
            },
            generation_type="content"
        )

    async def log_text_analysis_completion(self, texts_count: int, analysis_results: Dict):
        await self.log_analysis_completion(
            agent_name="TextAnalyzerAgent",
            analysis_result={
                "texts_analyzed": texts_count,
                "analysis_results": analysis_results,
                "analysis_depth": len(analysis_results.keys()) if isinstance(analysis_results, dict) else 1
            },
            analysis_type="text"
        )

    async def log_search_completion(self, query: str, results_count: int, search_metadata: Dict = None):
        await self.log_processing_completion(
            agent_name="SearchAgent",
            agent_role="검색 에이전트",
            process_type="search_operation",
            result_data={
                "query": query,
                "results_count": results_count,
                "search_success": results_count > 0,
                "metadata": search_metadata or {}
            }
        )

    async def log_filter_completion(self, filter_type: str, input_count: int, output_count: int, filter_criteria: Dict = None):
        await self.log_processing_completion(
            agent_name="FilterAgent",
            agent_role="필터링 에이전트",
            process_type=f"{filter_type}_filtering",
            result_data={
                "input_count": input_count,
                "output_count": output_count,
                "filter_ratio": output_count / input_count if input_count > 0 else 0,
                "filter_criteria": filter_criteria or {},
                "filter_effectiveness": "high" if output_count / input_count < 0.8 else "low"
            }
        )

    async def log_aggregation_completion(self, data_sources: List[str], aggregation_result: Dict):
        await self.log_processing_completion(
            agent_name="AggregationAgent", 
            agent_role="데이터 집계 에이전트",
            process_type="data_aggregation",
            result_data={
                "data_sources": data_sources,
                "sources_count": len(data_sources),
                "aggregation_result": aggregation_result,
                "aggregation_complexity": len(data_sources)
            }
        )

    async def log_workflow_completion(self, workflow_name: str, steps_completed: int, total_steps: int, workflow_result: Dict):
        await self.log_processing_completion(
            agent_name="WorkflowAgent",
            agent_role="워크플로우 관리자",
            process_type="workflow_execution",
            result_data={
                "workflow_name": workflow_name,
                "steps_completed": steps_completed,
                "total_steps": total_steps,
                "completion_rate": steps_completed / total_steps if total_steps > 0 else 0,
                "workflow_result": workflow_result,
                "workflow_success": steps_completed == total_steps
            }
        )

    def get_performance_summary(self, agent_name: str = None) -> Dict:
        if agent_name:
            return self.performance_tracker.get(agent_name, {})
        
        summary = {
            "total_agents": len(self.performance_tracker),
            "agents": {},
            "system_health": "healthy"
        }
        
        total_errors = 0
        total_requests = 0
        
        for agent, stats in self.performance_tracker.items():
            summary["agents"][agent] = stats.copy()
            total_errors += stats.get("total_errors", 0)
            total_requests += stats.get("total_requests", 0)
        
        if total_requests > 0:
            overall_error_rate = total_errors / total_requests
            if overall_error_rate > self.error_threshold:
                summary["system_health"] = "warning"
            if overall_error_rate > self.error_threshold * 2:
                summary["system_health"] = "critical"
        
        summary["overall_error_rate"] = total_errors / total_requests if total_requests > 0 else 0
        summary["total_requests"] = total_requests
        summary["total_errors"] = total_errors
        
        return summary

    def reset_performance_tracking(self, agent_name: str = None):
        if agent_name:
            if agent_name in self.performance_tracker:
                del self.performance_tracker[agent_name]
        else:
            self.performance_tracker.clear()
        
        self.logger.info(f"성능 추적 데이터 리셋: {agent_name or '전체'}")

    def get_system_health_report(self) -> Dict:
        summary = self.get_performance_summary()
        
        health_report = {
            "timestamp": time.time(),
            "overall_health": summary["system_health"],
            "total_agents": summary["total_agents"],
            "total_requests": summary["total_requests"],
            "error_rate": summary["overall_error_rate"],
            "agent_details": {},
            "recommendations": []
        }
        
        for agent_name, stats in summary["agents"].items():
            agent_error_rate = stats.get("total_errors", 0) / max(stats.get("total_requests", 1), 1)
            
            health_report["agent_details"][agent_name] = {
                "status": "healthy" if agent_error_rate < self.error_threshold else "warning",
                "error_rate": agent_error_rate,
                "avg_response_time": stats.get("avg_response_time", 0),
                "total_requests": stats.get("total_requests", 0)
            }
            
            if agent_error_rate > self.error_threshold:
                health_report["recommendations"].append(f"{agent_name} 에이전트의 에러율이 높습니다 ({agent_error_rate:.1%})")
        
        return health_report

    async def log_system_health_check(self):
        health_report = self.get_system_health_report()
        
        await self.log_agent_response(
            agent_name="SystemHealthMonitor",
            agent_role="시스템 건강도 모니터",
            task_description="시스템 건강도 점검",
            response_data=health_report,
            metadata={"health_check": True, "automated": True}
        )
        
        return health_report

    def configure_agent(self, agent_name: str, config: Dict):
        self.agent_configs[agent_name] = {**self.agent_configs.get(agent_name, {}), **config}
        self.logger.info(f"에이전트 {agent_name} 설정 업데이트: {config}")

    def get_agent_config(self, agent_name: str) -> Dict:
        return self.agent_configs.get(agent_name, {})

    def list_configured_agents(self) -> List[str]:
        return list(self.agent_configs.keys())
