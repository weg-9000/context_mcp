import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
import importlib

# 데이터베이스 추상화를 위한 인터페이스
class DatabaseInterface(ABC):
    """데이터베이스 인터페이스 - 다양한 DB 지원을 위한 추상화"""
    
    @abstractmethod
    def store_data(self, container_id: str, session_id: str, agent_name: str, data: dict) -> bool:
        """데이터 저장"""
        pass
    
    @abstractmethod
    def retrieve_data(self, container_id: str, session_id: str, agent_name: str = None) -> dict:
        """데이터 조회"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        pass

class DatabaseAdapterFactory:
    """데이터베이스 어댑터 팩토리 - 설정 기반 어댑터 생성"""
    
    _adapters = {}
    
    @classmethod
    def register_adapter(cls, adapter_name: str, adapter_class: type):
        """어댑터 등록"""
        cls._adapters[adapter_name] = adapter_class
    
    @classmethod
    def create_adapter(cls, config: Dict) -> DatabaseInterface:
        """설정 기반 어댑터 생성"""
        db_type = config.get('database_type', 'filesystem')
        
        if db_type in cls._adapters:
            return cls._adapters[db_type](config)
        
        # 동적 어댑터 로딩 시도
        adapter = cls._try_load_adapter(db_type, config)
        if adapter:
            return adapter
            
        # 폴백: 파일시스템 어댑터
        return FileSystemAdapter(config.get('filesystem_path', './agent_logs'))
    
    @classmethod
    def _try_load_adapter(cls, db_type: str, config: Dict) -> Optional[DatabaseInterface]:
        """동적 어댑터 로딩"""
        try:
            # 표준 어댑터 이름 매핑
            adapter_mapping = {
                'cosmos': 'CosmosDBAdapter',
                'cosmosdb': 'CosmosDBAdapter', 
                'postgresql': 'PostgreSQLAdapter',
                'postgres': 'PostgreSQLAdapter',
                'mysql': 'MySQLAdapter',
                'sqlite': 'SQLiteAdapter',
                'mongodb': 'MongoDBAdapter',
                'redis': 'RedisAdapter'
            }
            
            adapter_class_name = adapter_mapping.get(db_type.lower())
            if not adapter_class_name:
                return None
                
            # 동적 임포트 시도
            module_name = f"database_adapters.{db_type.lower()}_adapter"
            try:
                module = importlib.import_module(module_name)
                adapter_class = getattr(module, adapter_class_name)
                return adapter_class(config)
            except (ImportError, AttributeError):
                # 대안 경로 시도
                try:
                    from . import database_adapters
                    adapter_class = getattr(database_adapters, adapter_class_name)
                    return adapter_class(config)
                except (ImportError, AttributeError):
                    pass
                    
        except Exception as e:
            print(f"어댑터 로딩 실패 ({db_type}): {e}")
            
        return None
    
def _get_cosmos_client():
    try:
        from azure.cosmos import CosmosClient
        return CosmosClient
    except ImportError:
        return None
    
class ConfigurableCosmosDBAdapter(DatabaseInterface):
    """설정 가능한 Cosmos DB 어댑터"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.container = None
        self.update_func = None
        self.get_func = None
        self._connected = False
        
        # 다양한 Cosmos DB 설정 방식 지원
        self._initialize_cosmos_connection()
    
    def _initialize_cosmos_connection(self):
        """Cosmos DB 연결 초기화 - 다양한 설정 방식 지원"""
        cosmos_config = self.config.get('cosmos_config', {})
        
        try:
            # 방법 1: 직접 연결 정보 제공
            if 'endpoint' in cosmos_config and 'key' in cosmos_config:
                self._init_with_direct_config(cosmos_config)
            
            # 방법 2: 모듈 경로 제공
            elif 'connection_module' in cosmos_config:
                self._init_with_module_path(cosmos_config)
            
            # 방법 3: 팩토리 함수 제공
            elif 'connection_factory' in cosmos_config:
                self._init_with_factory(cosmos_config)
            
            # 방법 4: 기존 연결 객체 제공
            elif 'container_instance' in cosmos_config:
                self._init_with_instance(cosmos_config)
            
            # 방법 5: 기본 경로 시도 (기존 호환성)
            else:
                self._init_with_default_path()
                
        except Exception as e:
            print(f"Cosmos DB 초기화 실패: {e}")
            self._connected = False
    
    def _init_with_direct_config(self, cosmos_config: Dict):
            CosmosClient = _get_cosmos_client()
            if not CosmosClient:
                raise ImportError("azure-cosmos 패키지가 설치되지 않았습니다")
            
            client = CosmosClient(
                cosmos_config['endpoint'],
                cosmos_config['key']
            )
            
            database = client.get_database_client(cosmos_config['database_name'])
            self.container = database.get_container_client(cosmos_config['container_name'])
            
            # 기본 CRUD 함수 정의
            self.update_func = self._default_update
            self.get_func = self._default_get
            self._connected = True
            
    
    def _init_with_module_path(self, cosmos_config: Dict):
        """모듈 경로로 초기화"""
        try:
            module_path = cosmos_config['connection_module']
            module = importlib.import_module(module_path)
            
            # 컨테이너 객체 가져오기
            container_attr = cosmos_config.get('container_attribute', 'logging_container')
            self.container = getattr(module, container_attr)
            
            # CRUD 함수 가져오기
            update_func_name = cosmos_config.get('update_function', 'update_agent_logs_in_cosmos')
            get_func_name = cosmos_config.get('get_function', 'get_agent_logs_from_cosmos')
            
            self.update_func = getattr(module, update_func_name)
            self.get_func = getattr(module, get_func_name)
            self._connected = True
            
        except Exception as e:
            print(f"모듈 경로 Cosmos DB 연결 실패: {e}")
    
    def _init_with_factory(self, cosmos_config: Dict):
        """팩토리 함수로 초기화"""
        try:
            factory_func = cosmos_config['connection_factory']
            
            # 팩토리 함수 호출
            if callable(factory_func):
                connection_objects = factory_func(cosmos_config)
            else:
                # 문자열인 경우 동적 임포트
                module_path, func_name = factory_func.rsplit('.', 1)
                module = importlib.import_module(module_path)
                factory = getattr(module, func_name)
                connection_objects = factory(cosmos_config)
            
            self.container = connection_objects['container']
            self.update_func = connection_objects['update_func']
            self.get_func = connection_objects['get_func']
            self._connected = True
            
        except Exception as e:
            print(f"팩토리 Cosmos DB 연결 실패: {e}")
    
    def _init_with_instance(self, cosmos_config: Dict):
        """기존 연결 객체로 초기화"""
        try:
            self.container = cosmos_config['container_instance']
            self.update_func = cosmos_config.get('update_func', self._default_update)
            self.get_func = cosmos_config.get('get_func', self._default_get)
            self._connected = True
            
        except Exception as e:
            print(f"인스턴스 Cosmos DB 연결 실패: {e}")
    
    def _init_with_default_path(self):
        """기본 경로로 초기화 (기존 호환성)"""
        try:
            # 여러 기본 경로 시도
            default_paths = [
                ('db.cosmos_connection', 'logging_container'),
                ('...db.cosmos_connection', 'logging_container'),
                ('backend.app.db.cosmos_connection', 'logging_container')
            ]
            
            for module_path, container_name in default_paths:
                try:
                    module = importlib.import_module(module_path)
                    self.container = getattr(module, container_name)
                    
                    # 기본 함수들 임포트 시도
                    try:
                        db_utils = importlib.import_module(f"{module_path.rsplit('.', 1)[0]}.db_utils")
                        self.update_func = getattr(db_utils, 'update_agent_logs_in_cosmos')
                        self.get_func = getattr(db_utils, 'get_agent_logs_from_cosmos')
                    except:
                        self.update_func = self._default_update
                        self.get_func = self._default_get
                    
                    self._connected = True
                    break
                    
                except ImportError:
                    continue
                    
        except Exception as e:
            print(f"기본 경로 Cosmos DB 연결 실패: {e}")
    
    def _default_update(self, container, session_id: str, agent_name: str, data: dict):
        """기본 업데이트 함수"""
        try:
            # Cosmos DB 표준 upsert 사용
            item = {
                'id': f"{session_id}_{agent_name}",
                'session_id': session_id,
                'agent_name': agent_name,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            container.upsert_item(item)
        except Exception as e:
            print(f"Cosmos DB 기본 업데이트 실패: {e}")
            raise
    
    def _default_get(self, container, session_id: str, agent_name: str = None):
        """기본 조회 함수"""
        try:
            if agent_name:
                # 특정 에이전트 데이터 조회
                item_id = f"{session_id}_{agent_name}"
                try:
                    item = container.read_item(item_id, partition_key=session_id)
                    return item.get('data', {})
                except:
                    return {}
            else:
                # 세션의 모든 데이터 조회
                query = f"SELECT * FROM c WHERE c.session_id = '{session_id}'"
                items = list(container.query_items(query=query, enable_cross_partition_query=True))
                
                result = {'agent_outputs': {}}
                for item in items:
                    agent = item.get('agent_name', 'unknown')
                    result['agent_outputs'][agent] = item.get('data', {})
                
                return result
                
        except Exception as e:
            print(f"Cosmos DB 기본 조회 실패: {e}")
            return {}
    
    def store_data(self, container_id: str, session_id: str, agent_name: str, data: dict) -> bool:
        if not self._connected:
            return False
        try:
            self.update_func(self.container, session_id, agent_name, data)
            return True
        except Exception as e:
            print(f"Cosmos DB 저장 실패: {e}")
            return False
    
    def retrieve_data(self, container_id: str, session_id: str, agent_name: str = None) -> dict:
        if not self._connected:
            return {}
        try:
            if agent_name:
                return self.get_func(self.container, session_id, agent_name) or {}
            return self.get_func(self.container, session_id) or {}
        except Exception as e:
            print(f"Cosmos DB 조회 실패: {e}")
            return {}
    
    def is_connected(self) -> bool:
        return self._connected

class FileSystemAdapter(DatabaseInterface):
    """파일 시스템 어댑터 - 향상된 버전"""
    
    def __init__(self, base_path: str = "./agent_logs"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def store_data(self, container_id: str, session_id: str, agent_name: str, data: dict) -> bool:
        try:
            file_path = os.path.join(self.base_path, f"{session_id}_{agent_name}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"파일시스템 저장 실패: {e}")
            return False
    
    def retrieve_data(self, container_id: str, session_id: str, agent_name: str = None) -> dict:
        try:
            if agent_name:
                file_path = os.path.join(self.base_path, f"{session_id}_{agent_name}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            else:
                # 모든 파일 읽기
                all_data = {'agent_outputs': {}}
                for file in os.listdir(self.base_path):
                    if file.startswith(session_id) and file.endswith('.json'):
                        agent_name_from_file = file.replace(f"{session_id}_", "").replace(".json", "")
                        with open(os.path.join(self.base_path, file), 'r', encoding='utf-8') as f:
                            all_data['agent_outputs'][agent_name_from_file] = json.load(f)
                return all_data
            return {}
        except Exception as e:
            print(f"파일시스템 조회 실패: {e}")
            return {}
    
    def is_connected(self) -> bool:
        return os.path.exists(self.base_path)

# 기본 어댑터들 등록
DatabaseAdapterFactory.register_adapter('cosmos', ConfigurableCosmosDBAdapter)
DatabaseAdapterFactory.register_adapter('cosmosdb', ConfigurableCosmosDBAdapter)
DatabaseAdapterFactory.register_adapter('filesystem', FileSystemAdapter)

@dataclass
class AgentOutput:
    """에이전트 응답 데이터"""
    agent_name: str
    agent_role: str
    output_id: str
    timestamp: str
    task_description: str
    final_answer: str
    reasoning_process: str
    execution_steps: List[str]
    raw_input: Any
    raw_output: Any
    performance_metrics: Dict
    error_logs: List[Dict]
    info_data: Dict

    def get_info(self, key: str = None):
        """안전한 info 데이터 접근"""
        if key:
            return self.info_data.get(key)
        return self.info_data

    def set_info(self, key: str, value: Any):
        """안전한 info 데이터 설정"""
        if not hasattr(self, 'info_data') or self.info_data is None:
            self.info_data = {}
        self.info_data[key] = value

@dataclass
class AgentInfo:
    """에이전트 정보 데이터"""
    agent_name: str
    info_id: str
    timestamp: str
    info_type: str
    info_content: Dict
    metadata: Dict
    info_data: Dict

class AgentOutputManager:
    """에이전트 응답 전용 관리 시스템 - 범용성 강화"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.current_session_id = self._generate_session_id()
        self.outputs = []
        
        # 설정 가능한 옵션들
        self.max_memory_cache = self.config.get('max_memory_cache', 100)
        self.cache_cleanup_ratio = self.config.get('cache_cleanup_ratio', 0.5)
        
        # 팩토리를 통한 데이터베이스 어댑터 생성
        self.db_adapter = DatabaseAdapterFactory.create_adapter(self.config)
        
    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def store_agent_output(self,
                          agent_name: str,
                          agent_role: str,
                          task_description: str,
                          final_answer: str,
                          reasoning_process: str = "",
                          execution_steps: List[str] = None,
                          raw_input: Any = None,
                          raw_output: Any = None,
                          performance_metrics: Dict = None,
                          error_logs: List[Dict] = None,
                          info_data: Dict = None) -> str:
        """에이전트 응답 저장 (데이터베이스 + 메모리 캐싱)"""
        
        output_id = f"{agent_name}_{int(time.time() * 1000000)}"
        
        agent_output = AgentOutput(
            agent_name=agent_name,
            agent_role=agent_role,
            output_id=output_id,
            timestamp=datetime.now().isoformat(),
            task_description=task_description,
            final_answer=final_answer,
            reasoning_process=reasoning_process,
            execution_steps=execution_steps or [],
            raw_input=self._safe_copy(raw_input),
            raw_output=self._safe_copy(raw_output),
            performance_metrics=performance_metrics or {},
            error_logs=error_logs or [],
            info_data=info_data or {}
        )

        # 메모리에 저장 (로컬 캐싱)
        self.outputs.append(agent_output)

        # 데이터베이스에 저장
        try:
            output_data = asdict(agent_output)
            success = self.db_adapter.store_data(
                "logging_container",
                self.current_session_id,
                agent_name,
                output_data
            )
            
            if success:
                print(f"📦 {agent_name} 응답을 데이터베이스에 저장: {output_id}")
            else:
                print(f"⚠️ 데이터베이스 저장 실패, 로컬 캐시만 사용: {agent_name}")
                
        except Exception as e:
            print(f"❌ 데이터베이스 저장 실패, 로컬에만 저장됨: {e}")

        # 메모리 관리
        self._manage_memory_cache()
        
        return output_id

    def _safe_copy(self, data: Any) -> Any:
        """안전한 데이터 복사"""
        try:
            if data is None:
                return None
            if isinstance(data, (str, int, float, bool)):
                return data
            if isinstance(data, (list, tuple)):
                return [self._safe_copy(item) for item in data]
            if isinstance(data, dict):
                return {key: self._safe_copy(value) for key, value in data.items()}
            return str(data)  # 복잡한 객체는 문자열로 변환
        except:
            return str(data)

    def _manage_memory_cache(self):
        """메모리 캐시 관리"""
        try:
            outputs_count = len(self.outputs)
            if outputs_count > self.max_memory_cache:
                # 가장 오래된 항목 일부 제거
                keep_count = int(self.max_memory_cache * self.cache_cleanup_ratio)
                self.outputs = self.outputs[-keep_count:]
                print(f"🔄 메모리 캐시 크기 조정: {len(self.outputs)}개 항목 유지")
        except Exception as e:
            print(f"❌ 메모리 캐싱 실패: {e}")

    def get_all_outputs(self, exclude_agent: str = None) -> List[Dict]:
        """모든 에이전트 응답 조회"""
        try:
            # 데이터베이스에서 먼저 조회
            db_logs = self.db_adapter.retrieve_data(
                "logging_container", 
                self.current_session_id
            )
            
            if db_logs:
                all_outputs = []
                for agent, outputs in db_logs.get("agent_outputs", {}).items():
                    if exclude_agent is None or agent != exclude_agent:
                        if isinstance(outputs, list):
                            all_outputs.extend(outputs)
                        else:
                            all_outputs.append(outputs)
                
                # 타임스탬프로 정렬
                return sorted(all_outputs, key=lambda x: x.get('timestamp', ''))
                
        except Exception as e:
            print(f"⚠️ 데이터베이스 로그 조회 실패, 로컬 캐시 사용: {e}")

        # 로컬 캐시에서 조회 (폴백)
        all_outputs = []
        for output in self.outputs:
            if exclude_agent is None or output.agent_name != exclude_agent:
                all_outputs.append(asdict(output))
        
        return sorted(all_outputs, key=lambda x: x.get('timestamp', ''))

    def get_agent_output(self, agent_name: str, latest: bool = True) -> Optional[Dict]:
        """특정 에이전트의 응답 조회"""
        try:
            # 데이터베이스에서 먼저 조회
            agent_outputs = self.db_adapter.retrieve_data(
                "logging_container",
                self.current_session_id,
                agent_name
            )
            
            if agent_outputs:
                if isinstance(agent_outputs, list):
                    if latest:
                        return sorted(agent_outputs, key=lambda x: x.get('timestamp', ''), reverse=True)[0]
                    else:
                        return agent_outputs
                else:
                    return agent_outputs
                    
        except Exception as e:
            print(f"⚠️ 데이터베이스 에이전트 로그 조회 실패, 로컬 캐시 사용: {e}")

        # 로컬 캐시에서 조회 (폴백)
        agent_outputs = [
            asdict(output) for output in self.outputs
            if output.agent_name == agent_name
        ]
        
        if not agent_outputs:
            return None
            
        if latest:
            return sorted(agent_outputs, key=lambda x: x.get('timestamp', ''), reverse=True)[0]
        else:
            return agent_outputs

    def store_agent_info(self, agent_name: str, info_type: str, info_content: Dict, metadata: Dict = None) -> str:
        """에이전트 정보 저장"""
        info_id = f"{agent_name}_info_{int(time.time() * 1000000)}"
        
        agent_info = {
            "agent_name": agent_name,
            "info_id": info_id,
            "timestamp": datetime.now().isoformat(),
            "info_type": info_type,
            "info_content": info_content,
            "metadata": metadata or {},
            "info_data": {}
        }

        # 데이터베이스에 저장
        try:
            self.db_adapter.store_data(
                "logging_container",
                self.current_session_id,
                f"{agent_name}_info",
                agent_info
            )
        except Exception as e:
            print(f"❌ 에이전트 정보 저장 실패: {e}")

        return info_id

    def get_agent_info(self, agent_name: str = None, info_type: str = None, latest: bool = True) -> List[Dict]:
        """에이전트 정보 조회"""
        try:
            # 데이터베이스에서 조회
            db_logs = self.db_adapter.retrieve_data("logging_container", self.current_session_id)
            
            if not db_logs:
                return []

            agent_info = []
            
            # 정보 필터링
            for agent, outputs in db_logs.get("agent_outputs", {}).items():
                if agent.endswith("_info"):  # 정보 항목 식별자
                    agent_base_name = agent.replace("_info", "")
                    if agent_name and agent_base_name != agent_name:
                        continue
                    
                    if isinstance(outputs, list):
                        for info in outputs:
                            if info_type and info.get("info_type") != info_type:
                                continue
                            agent_info.append(info)
                    else:
                        if not info_type or outputs.get("info_type") == info_type:
                            agent_info.append(outputs)

            if latest and agent_info:
                agent_info = sorted(agent_info, key=lambda x: x.get("timestamp", ""), reverse=True)
                if agent_name and info_type:
                    # 특정 에이전트의 특정 타입 정보 중 최신
                    return [agent_info[0]]
                    
            return agent_info
            
        except Exception as e:
            print(f"❌ 에이전트 정보 조회 실패: {e}")
            return []

    def get_all_info(self) -> List[Dict]:
        """모든 정보 조회"""
        return self.get_agent_info(agent_name=None, info_type=None, latest=False)

class AgentDecisionLogger:
    """범용 에이전트 로거 (완전히 설정 기반)"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.current_session_id = self._generate_session_id()
        self.output_manager = AgentOutputManager(config)

    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def log_agent_real_output(self,
                             agent_name: str,
                             agent_role: str,
                             task_description: str,
                             final_answer: str,
                             reasoning_process: str = "",
                             execution_steps: List[str] = None,
                             raw_input: Any = None,
                             raw_output: Any = None,
                             performance_metrics: Dict = None,
                             error_logs: List[Dict] = None,
                             info_data: Dict = None) -> str:
        """에이전트 응답 로깅"""
        return self.output_manager.store_agent_output(
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

    def log_agent_info(self,
                      agent_name: str,
                      info_type: str,
                      info_content: Dict,
                      metadata: Dict = None) -> str:
        """에이전트 정보 로깅"""
        return self.output_manager.store_agent_info(
            agent_name=agent_name,
            info_type=info_type,
            info_content=info_content,
            metadata=metadata
        )

    def get_agent_info(self, agent_name: str = None, info_type: str = None, latest: bool = True) -> List[Dict]:
        """에이전트 정보 조회"""
        return self.output_manager.get_agent_info(agent_name, info_type, latest)

    def get_all_info(self) -> List[Dict]:
        """모든 정보 조회"""
        return self.output_manager.get_all_info()

    def get_all_previous_results(self, current_agent: str) -> List[Dict]:
        """모든 이전 응답 조회"""
        return self.output_manager.get_all_outputs(exclude_agent=current_agent)

    def get_previous_agent_result(self, agent_name: str, latest: bool = True) -> Optional[Dict]:
        """이전 에이전트 응답 조회"""
        return self.output_manager.get_agent_output(agent_name, latest)

    def get_learning_insights(self, target_agent: str) -> Dict:
        """학습 인사이트 생성 (개선됨)"""
        all_outputs = self.output_manager.get_all_outputs()
        
        if not all_outputs:
            return {
                "insights": "이전 에이전트 응답이 없습니다.",
                "patterns": [],
                "recommendations": []
            }

        # 패턴 분석 개선
        patterns = self._analyze_output_patterns(all_outputs)
        recommendations = self._generate_recommendations(patterns, target_agent)

        return {
            "target_agent": target_agent,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_outputs_analyzed": len(all_outputs),
            "patterns": patterns,
            "recommendations": recommendations,
            "insights": self._extract_insights(all_outputs, target_agent),
            "system_health": self._analyze_system_health(all_outputs)
        }

    def _analyze_output_patterns(self, outputs: List[Dict]) -> List[Dict]:
        """응답 패턴 분석 (개선됨)"""
        # 에이전트별 응답 그룹화
        agent_groups = {}
        for output in outputs:
            agent_name = output.get("agent_name", "unknown")
            if agent_name not in agent_groups:
                agent_groups[agent_name] = []
            agent_groups[agent_name].append(output)

        patterns = []

        # 각 에이전트별 패턴 분석
        for agent_name, agent_outputs in agent_groups.items():
            if not agent_outputs:
                continue

            # 기본 통계
            final_answers = [output.get("final_answer", "") for output in agent_outputs]
            avg_length = sum(len(ans) for ans in final_answers) / len(final_answers)
            
            # 성능 메트릭 분석
            performance_scores = []
            for output in agent_outputs:
                metrics = output.get("performance_metrics", {})
                if "confidence_score" in metrics:
                    performance_scores.append(metrics["confidence_score"])
                    
            # 에러 분석
            error_count = sum(1 for output in agent_outputs if output.get("error_logs"))
            
            pattern = {
                "agent": agent_name,
                "response_count": len(agent_outputs),
                "avg_response_length": avg_length,
                "response_pattern": "text" if avg_length > 0 else "structured",
                "avg_confidence": sum(performance_scores) / len(performance_scores) if performance_scores else 0,
                "error_rate": error_count / len(agent_outputs) if agent_outputs else 0,
                "activity_trend": self._analyze_activity_trend(agent_outputs)
            }
            patterns.append(pattern)

        return patterns

    def _analyze_activity_trend(self, agent_outputs: List[Dict]) -> str:
        """활동 트렌드 분석"""
        if len(agent_outputs) < 2:
            return "insufficient_data"
            
        # 시간순 정렬
        sorted_outputs = sorted(agent_outputs, key=lambda x: x.get('timestamp', ''))
        
        # 최근 활동과 과거 활동 비교 (간단한 구현)
        recent_count = len([o for o in sorted_outputs[-5:]])  # 최근 5개
        total_count = len(sorted_outputs)
        
        if recent_count / total_count > 0.7:
            return "increasing"
        elif recent_count / total_count < 0.3:
            return "decreasing"
        else:
            return "stable"

    def _generate_recommendations(self, patterns: List[Dict], target_agent: str) -> List[str]:
        """추천 생성 (개선됨)"""
        if not patterns:
            return ["분석할 패턴이 없습니다."]

        recommendations = []

        # 타겟 에이전트 패턴
        target_pattern = None
        for pattern in patterns:
            if pattern["agent"] == target_agent:
                target_pattern = pattern
                break

        if target_pattern:
            # 응답 길이 관련 추천
            if target_pattern["avg_response_length"] > 1000:
                recommendations.append(f"{target_agent}의 응답이 매우 깁니다. 보다 간결한 응답을 고려하세요.")
            elif target_pattern["avg_response_length"] < 50:
                recommendations.append(f"{target_agent}의 응답이 매우 짧습니다. 보다 상세한 응답이 필요할 수 있습니다.")
                
            # 신뢰도 관련 추천
            if target_pattern["avg_confidence"] < 0.5:
                recommendations.append(f"{target_agent}의 평균 신뢰도가 낮습니다. 모델 파라미터나 프롬프트 개선을 검토하세요.")
                
            # 에러율 관련 추천
            if target_pattern["error_rate"] > 0.1:
                recommendations.append(f"{target_agent}의 에러율이 {target_pattern['error_rate']:.1%}입니다. 에러 처리 로직을 검토하세요.")
                
            # 활동 트렌드 관련 추천
            if target_pattern["activity_trend"] == "decreasing":
                recommendations.append(f"{target_agent}의 활동이 감소하고 있습니다. 사용 패턴을 검토하세요.")

        return recommendations

    def _extract_insights(self, outputs: List[Dict], target_agent: str) -> List[str]:
        """인사이트 추출 (개선됨)"""
        insights = []

        # 타겟 에이전트 출력만 필터링
        target_outputs = [output for output in outputs if output.get("agent_name") == target_agent]

        if not target_outputs:
            insights.append(f"{target_agent}의 이전 응답이 없습니다.")
            return insights

        # 시간순 정렬
        target_outputs.sort(key=lambda x: x.get("timestamp", ""))

        # 최신 응답
        latest_output = target_outputs[-1]
        latest_task = latest_output.get("task_description", "")
        insights.append(f"{target_agent}의 최근 작업: {latest_task}")

        # 성능 메트릭스 분석
        metrics = [output.get("performance_metrics", {}) for output in target_outputs]
        if metrics:
            # 평균 응답 길이
            response_lengths = [m.get("response_length", 0) for m in metrics if "response_length" in m]
            if response_lengths:
                avg_length = sum(response_lengths) / len(response_lengths)
                insights.append(f"평균 응답 길이: {avg_length:.1f}")
                
            # 신뢰도 점수
            confidence_scores = [m.get("confidence_score", 0) for m in metrics if "confidence_score" in m]
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                insights.append(f"평균 신뢰도: {avg_confidence:.2f}")

        # 작업 유형 분석
        task_types = {}
        for output in target_outputs:
            task = output.get("task_description", "")
            # 간단한 키워드 기반 분류
            if "분석" in task:
                task_types["분석"] = task_types.get("분석", 0) + 1
            elif "생성" in task:
                task_types["생성"] = task_types.get("생성", 0) + 1
            elif "검색" in task:
                task_types["검색"] = task_types.get("검색", 0) + 1
            else:
                task_types["기타"] = task_types.get("기타", 0) + 1
                
        if task_types:
            most_common = max(task_types, key=task_types.get)
            insights.append(f"가장 빈번한 작업 유형: {most_common} ({task_types[most_common]}회)")

        return insights

    def _analyze_system_health(self, outputs: List[Dict]) -> Dict:
        """시스템 건강도 분석"""
        if not outputs:
            return {"status": "no_data", "score": 0}
            
        total_outputs = len(outputs)
        error_count = sum(1 for output in outputs if output.get("error_logs"))
        error_rate = error_count / total_outputs
        
        # 최근 활동 분석
        recent_outputs = [o for o in outputs if o.get("timestamp", "") > datetime.now().replace(hour=0, minute=0, second=0).isoformat()]
        recent_activity = len(recent_outputs) / total_outputs if total_outputs > 0 else 0
        
        # 건강도 점수 계산 (0-1)
        health_score = (1 - error_rate) * 0.7 + recent_activity * 0.3
        
        return {
            "status": "healthy" if health_score > 0.8 else "warning" if health_score > 0.5 else "critical",
            "score": health_score,
            "total_outputs": total_outputs,
            "error_rate": error_rate,
            "recent_activity_ratio": recent_activity
        }

    def log_agent_decision(self, agent_name: str, agent_role: str, input_data: Dict,
                          decision_process: Dict, output_result: Dict, reasoning: str,
                          confidence_score: float = 0.8, context: Dict = None,
                          performance_metrics: Dict = None) -> str:
        """에이전트 결정 로깅 (이전 버전 호환성 유지)"""
        metrics = performance_metrics or {}
        metrics["confidence_score"] = confidence_score

        return self.log_agent_real_output(
            agent_name=agent_name,
            agent_role=agent_role,
            task_description=f"결정: {list(decision_process.keys())[0] if decision_process else ''}",
            final_answer=output_result.get("answer", str(output_result)),
            reasoning_process=reasoning,
            raw_input=input_data,
            raw_output=output_result,
            performance_metrics=metrics,
            info_data=context
        )

    def log_agent_interaction(self,
                             source_agent: str,
                             target_agent: str,
                             interaction_type: str,
                             data_transferred: Dict,
                             success: bool = True) -> str:
        """에이전트 간 상호작용 로깅"""
        return self.log_agent_real_output(
            agent_name=f"{source_agent}_to_{target_agent}",
            agent_role="상호작용",
            task_description=f"{interaction_type} 상호작용",
            final_answer=f"성공: {success}",
            raw_input={
                "source": source_agent,
                "target": target_agent,
                "type": interaction_type
            },
            raw_output=data_transferred,
            performance_metrics={
                "success": success,
                "interaction_type": interaction_type
            }
        )

# 팩토리 함수들 (설정 기반으로 개선)
def get_agent_logger(config: Dict = None) -> AgentDecisionLogger:
    """전역 에이전트 로거 인스턴스 반환 (설정 지원)"""
    # 싱글톤 패턴
    if not hasattr(get_agent_logger, "instance"):
        get_agent_logger.instance = AgentDecisionLogger(config)
    return get_agent_logger.instance

def get_real_output_manager(config: Dict = None) -> AgentOutputManager:
    """전역 에이전트 출력 관리자 인스턴스 반환 (설정 지원)"""
    # 싱글톤 패턴
    if not hasattr(get_real_output_manager, "instance"):
        get_real_output_manager.instance = AgentOutputManager(config)
    return get_real_output_manager.instance

def get_complete_data_manager(config: Dict = None) -> AgentOutputManager:
    """데이터베이스 연결된 에이전트 출력 관리자 인스턴스 반환 (설정 지원)"""
    # 싱글톤 패턴
    if not hasattr(get_complete_data_manager, "instance"):
        get_complete_data_manager.instance = AgentOutputManager(config)
    return get_complete_data_manager.instance

# 편의 함수들
def log_agent_decision(agent_name: str, agent_role: str = None, input_data: Dict = None,
                      decision_process: Dict = None, output_result: Dict = None, reasoning: str = "",
                      confidence_score: float = 0.8, context: Dict = None,
                      performance_metrics: Dict = None, config: Dict = None) -> str:
    """에이전트 결정 로깅 편의 함수 (설정 지원)"""
    logger = get_agent_logger(config)
    return logger.log_agent_decision(
        agent_name=agent_name,
        agent_role=agent_role or "에이전트",
        input_data=input_data or {},
        decision_process=decision_process or {"default": []},
        output_result=output_result or {},
        reasoning=reasoning,
        confidence_score=confidence_score,
        context=context,
        performance_metrics=performance_metrics
    )
