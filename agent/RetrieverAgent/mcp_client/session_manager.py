import asyncio
import logging
import weakref
from typing import Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .config import MCPServerConfig

class SafeMCPSession:
    """안전한 MCP 세션 래퍼"""
    
    def __init__(self, session: 'ClientSession', cleanup_callback):
        self.session = session
        self.cleanup_callback = cleanup_callback
        self.is_active = True
        self._lock = asyncio.Lock()
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """도구 호출"""
        async with self._lock:
            if not self.is_active:
                raise RuntimeError("세션이 비활성화됨")
            return await self.session.call_tool(name, arguments)
    
    async def list_tools(self) -> Any:
        """도구 목록 조회"""
        async with self._lock:
            if not self.is_active:
                raise RuntimeError("세션이 비활성화됨")
            return await self.session.list_tools()
    
    async def close(self):
        """세션 종료"""
        async with self._lock:
            if self.is_active:
                self.is_active = False
                if self.cleanup_callback:
                    await self.cleanup_callback()

class ImprovedMCPSessionManager:
    """개선된 MCP 세션 관리자"""
    
    def __init__(self):
        self.sessions: Dict[str, SafeMCPSession] = {}
        self.connection_contexts: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.ImprovedMCPSessionManager")
        self._cleanup_lock = asyncio.Lock()
        self._session_locks: Dict[str, asyncio.Lock] = {}
        
        # 약한 참조를 사용하여 메모리 누수 방지
        self._finalizer = weakref.finalize(self, self._cleanup_all_sync)
    
    async def create_session(self, server_name: str, config: MCPServerConfig) -> bool:
        """안전한 세션 생성"""
        if not MCP_AVAILABLE:
            self.logger.error("MCP 라이브러리가 사용 불가능합니다.")
            return False
        
        # 서버별 락 생성
        if server_name not in self._session_locks:
            self._session_locks[server_name] = asyncio.Lock()
        
        async with self._session_locks[server_name]:
            try:
                # 기존 세션이 있다면 정리
                if server_name in self.sessions:
                    await self._cleanup_session_internal(server_name)
                
                # 새 세션 생성
                success = await self._create_session_with_timeout(server_name, config)
                
                if success:
                    self.logger.info(f"MCP 세션 생성 성공: {server_name}")
                else:
                    self.logger.error(f"MCP 세션 생성 실패: {server_name}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"세션 생성 중 예외 발생: {server_name}, 오류: {e}")
                return False
    
    async def _create_session_with_timeout(self, server_name: str, config: MCPServerConfig) -> bool:
        """타임아웃을 적용한 세션 생성"""
        try:
            # 타임아웃 설정 (기본 30초)
            timeout = getattr(config, 'timeout', 30.0)
            
            # 세션 생성을 별도 태스크로 실행
            creation_task = asyncio.create_task(
                self._create_session_impl(server_name, config)
            )
            
            # 타임아웃 적용
            return await asyncio.wait_for(creation_task, timeout=timeout)
            
        except asyncio.TimeoutError:
            self.logger.error(f"세션 생성 타임아웃: {server_name}")
            return False
        except Exception as e:
            self.logger.error(f"세션 생성 오류: {server_name}, {e}")
            return False
    
    async def _create_session_impl(self, server_name: str, config: MCPServerConfig) -> bool:
        """실제 세션 생성 구현"""
        context_manager = None
        session = None
        
        try:
            if config.transport == "stdio":
                # STDIO 서버 매개변수 생성
                from mcp import StdioServerParameters
                
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env=config.env
                )
                
                # 컨텍스트 매니저 생성 및 진입
                context_manager = stdio_client(server_params)
                read_stream, write_stream = await context_manager.__aenter__()
                
                # 세션 생성 및 초기화
                session = ClientSession(read_stream, write_stream)
                await session.initialize()
                
                # 정리 콜백 함수 정의
                async def cleanup_callback():
                    try:
                        if context_manager:
                            await context_manager.__aexit__(None, None, None)
                    except Exception as e:
                        self.logger.warning(f"컨텍스트 정리 중 오류: {e}")
                
                # 안전한 세션 래퍼 생성
                safe_session = SafeMCPSession(session, cleanup_callback)
                
                # 세션 저장
                self.sessions[server_name] = safe_session
                self.connection_contexts[server_name] = context_manager
                
                return True
                
            else:
                raise ValueError(f"지원하지 않는 전송 방식: {config.transport}")
                
        except Exception as e:
            self.logger.error(f"세션 구현 생성 실패: {server_name}, {e}")
            
            # 실패 시 정리
            if context_manager:
                try:
                    await context_manager.__aexit__(type(e), e, e.__traceback__)
                except Exception as cleanup_error:
                    self.logger.warning(f"실패 시 정리 오류: {cleanup_error}")
            
            return False
    
    async def get_session(self, server_name: str) -> Optional[SafeMCPSession]:
        """세션 조회"""
        return self.sessions.get(server_name)
    
    async def cleanup_session(self, server_name: str):
        """특정 세션 정리"""
        if server_name in self._session_locks:
            async with self._session_locks[server_name]:
                await self._cleanup_session_internal(server_name)
        else:
            await self._cleanup_session_internal(server_name)
    
    async def _cleanup_session_internal(self, server_name: str):
        """내부 세션 정리 로직"""
        try:
            # 세션 정리
            if server_name in self.sessions:
                session = self.sessions[server_name]
                await session.close()
                del self.sessions[server_name]
            
            # 컨텍스트 정리
            if server_name in self.connection_contexts:
                del self.connection_contexts[server_name]
            
            self.logger.info(f"세션 정리 완료: {server_name}")
            
        except Exception as e:
            self.logger.error(f"세션 정리 실패: {server_name}, 오류: {e}")
    
    async def cleanup_all(self):
        """모든 세션 정리"""
        async with self._cleanup_lock:
            cleanup_tasks = []
            
            for server_name in list(self.sessions.keys()):
                # 각 세션 정리를 별도 태스크로 실행
                task = asyncio.create_task(
                    self._cleanup_session_internal(server_name)
                )
                cleanup_tasks.append(task)
            
            # 모든 정리 태스크 완료 대기 (타임아웃 적용)
            if cleanup_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*cleanup_tasks, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("일부 세션 정리가 타임아웃됨")
                    
                    # 미완료 태스크 취소
                    for task in cleanup_tasks:
                        if not task.done():
                            task.cancel()
    
    def _cleanup_all_sync(self):
        """동기 방식 정리 (finalizer용)"""
        try:
            # 현재 실행 중인 이벤트 루프가 있는지 확인
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                # 비동기 정리 태스크 생성
                asyncio.create_task(self.cleanup_all())
        except RuntimeError:
            # 이벤트 루프가 없거나 종료된 경우 로그만 출력
            print("MCP 세션 매니저 정리됨")

    @asynccontextmanager
    async def session_context(self, server_name: str, config: MCPServerConfig):
        """세션 컨텍스트 매니저"""
        success = await self.create_session(server_name, config)
        if not success:
            raise RuntimeError(f"세션 생성 실패: {server_name}")
        
        try:
            session = await self.get_session(server_name)
            yield session
        finally:
            await self.cleanup_session(server_name)

    def __del__(self):
        """소멸자"""
        # finalizer가 정리를 담당하므로 별도 작업 불필요
        pass