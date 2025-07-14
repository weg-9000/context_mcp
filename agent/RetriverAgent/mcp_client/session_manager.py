import logging
from typing import Dict, Optional, Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .config import MCPServerConfig

class MCPSessionManager:
    """MCP 서버 세션들을 관리하는 클래스"""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.connections: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.MCPSessionManager")
    
    async def create_session(self, server_name: str, config: MCPServerConfig) -> bool:
        """새로운 MCP 서버 세션 생성"""
        if not MCP_AVAILABLE:
            self.logger.error("MCP 라이브러리가 사용 불가능합니다.")
            return False
            
        try:
            if config.transport == "stdio":
                # STDIO 전송 방식
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env=config.env
                )
                
                # stdio_client를 사용하여 연결
                read, write = await stdio_client(server_params).__aenter__()
                session = ClientSession(read, write)
                await session.initialize()
                
                self.connections[server_name] = (read, write)
                self.sessions[server_name] = session
                
            elif config.transport == "http":
                # HTTP 전송 방식 (향후 구현)
                raise NotImplementedError("HTTP 전송은 아직 구현되지 않았습니다.")
                
            else:
                raise ValueError(f"지원하지 않는 전송 방식: {config.transport}")
                
            self.logger.info(f"MCP 세션 생성 완료: {server_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"MCP 세션 생성 실패: {server_name}, 오류: {e}")
            return False
    
    async def get_session(self, server_name: str) -> Optional[ClientSession]:
        """서버 세션 조회"""
        return self.sessions.get(server_name)
    
    async def cleanup_session(self, server_name: str):
        """특정 세션 정리"""
        if server_name in self.sessions:
            try:
                session = self.sessions[server_name]
                # 세션 정리 로직 (필요시)
                del self.sessions[server_name]
                
                if server_name in self.connections:
                    # 연결 정리
                    del self.connections[server_name]
                    
                self.logger.info(f"MCP 세션 정리 완료: {server_name}")
            except Exception as e:
                self.logger.error(f"MCP 세션 정리 실패: {server_name}, 오류: {e}")
    
    async def cleanup_all(self):
        """모든 세션 정리"""
        for server_name in list(self.sessions.keys()):
            await self.cleanup_session(server_name)