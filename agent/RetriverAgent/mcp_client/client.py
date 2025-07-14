import logging
from typing import Dict, Any, List

try:
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP 라이브러리가 설치되지 않았습니다. pip install mcp langchain-mcp-adapters")

from .session_manager import MCPSessionManager
from .config import MCPServerConfig

class MCPClient:
    """MCP 서버와의 통신을 담당하는 클라이언트"""
    
    def __init__(self, server_configs: Dict[str, MCPServerConfig]):
        self.server_configs = server_configs
        self.session_manager = MCPSessionManager()
        self.tools_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.MCPClient")
        self.is_initialized = False
        
    async def initialize(self):
        """MCP 클라이언트 초기화"""
        if not MCP_AVAILABLE:
            self.logger.error("MCP 라이브러리가 사용 불가능합니다.")
            return False
            
        try:
            # 각 서버별 세션 초기화
            for server_name, config in self.server_configs.items():
                await self.session_manager.create_session(server_name, config)
                
            # 모든 서버의 도구 로드
            await self._load_all_tools()
            
            self.is_initialized = True
            self.logger.info(f"MCP 클라이언트 초기화 완료: {len(self.server_configs)}개 서버")
            return True
            
        except Exception as e:
            self.logger.error(f"MCP 클라이언트 초기화 실패: {e}")
            return False
    
    async def _load_all_tools(self):
        """모든 MCP 서버의 도구 로드"""
        for server_name in self.server_configs.keys():
            session = await self.session_manager.get_session(server_name)
            if session:
                try:
                    tools = await load_mcp_tools(session)
                    self.tools_cache[server_name] = tools
                    self.logger.info(f"{server_name} 서버에서 {len(tools)}개 도구 로드")
                except Exception as e:
                    self.logger.error(f"{server_name} 서버 도구 로드 실패: {e}")
    
    async def call_tool(self, server_name: str, tool_name: str, params: Dict[str, Any]) -> Any:
        """특정 MCP 서버의 도구 호출"""
        if not self.is_initialized:
            await self.initialize()
            
        if server_name not in self.tools_cache:
            raise ValueError(f"서버 '{server_name}'를 찾을 수 없습니다.")
            
        tools = self.tools_cache[server_name]
        
        # 도구 이름으로 찾기
        target_tool = None
        for tool in tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                target_tool = tool
                break
                
        if target_tool is None:
            raise ValueError(f"도구 '{tool_name}'을 서버 '{server_name}'에서 찾을 수 없습니다.")
            
        try:
            result = await target_tool.ainvoke(params)
            self.logger.debug(f"도구 호출 성공: {server_name}.{tool_name}")
            return result
        except Exception as e:
            self.logger.error(f"도구 호출 실패: {server_name}.{tool_name}, 오류: {e}")
            raise
    
    async def get_available_tools(self, server_name: str = None) -> Dict[str, List[str]]:
        """사용 가능한 도구 목록 조회"""
        if server_name:
            tools = self.tools_cache.get(server_name, [])
            return {server_name: [tool.name for tool in tools if hasattr(tool, 'name')]}
        else:
            result = {}
            for srv_name, tools in self.tools_cache.items():
                result[srv_name] = [tool.name for tool in tools if hasattr(tool, 'name')]
            return result
    
    async def cleanup(self):
        """리소스 정리"""
        await self.session_manager.cleanup_all()
        self.tools_cache.clear()
        self.is_initialized = False
        self.logger.info("MCP 클라이언트 정리 완료")