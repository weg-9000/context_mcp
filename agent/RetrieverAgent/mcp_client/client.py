import asyncio
import logging
import os
from typing import Dict, Any, List

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.client.stdio import StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .config import MCPServerConfig, DEFAULT_MCP_SERVERS

class MCPClient:
    """MCP 서버와 통신하는 클라이언트"""

    def __init__(self, server_configs: Dict[str, MCPServerConfig] = None):
        self.server_configs = server_configs or DEFAULT_MCP_SERVERS
        self.sessions: Dict[str, ClientSession] = {}
        self.logger = logging.getLogger(f"{__name__}.MCPClient")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """MCP 클라이언트 초기화"""
        if not MCP_AVAILABLE:
            self.logger.warning("MCP 라이브러리가 없습니다. 폴백 모드로 작동합니다.")
            return False

        try:
            for server_name, config in self.server_configs.items():
                if config.transport == "stdio":
                    await self._initialize_stdio_server(server_name, config)
            self.is_initialized = True
            self.logger.info(f"MCP 클라이언트 초기화 완료. 서버: {list(self.server_configs.keys())}")
            return True
        except Exception as e:
            self.logger.error(f"MCP 클라이언트 초기화 실패: {e}")
            return False

    async def _initialize_stdio_server(self, server_name: str, config: MCPServerConfig):
        """STDIO 서버 초기화"""
        try:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env={**os.environ, **config.env} if config.env else None
            )

            read, write = await stdio_client(server_params).__aenter__()
            session = ClientSession(read, write)
            await session.initialize()
            self.sessions[server_name] = session
            self.logger.info(f"MCP 서버 '{server_name}' 연결 성공")
        except Exception as e:
            self.logger.error(f"MCP 서버 '{server_name}' 연결 실패: {e}")

    async def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환"""
        if not self.is_initialized:
            return []
        try:
            available_tools = []
            for server_name, session in self.sessions.items():
                tools_response = await session.list_tools()
                available_tools.extend([tool.name for tool in tools_response.tools])
            return available_tools
        except Exception as e:
            self.logger.error(f"도구 목록 조회 실패: {e}")
            return []

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 서버의 도구 호출"""
        if not self.is_initialized or server_name not in self.sessions:
            return await self._fallback_tool_call(tool_name, arguments)

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, arguments)
            self.logger.info(f"도구 호출 성공: {tool_name}")
            return {"result": result.content, "status": "success"}
        except Exception as e:
            self.logger.error(f"도구 호출 실패 ({tool_name}): {e}")
            return await self._fallback_tool_call(tool_name, arguments)

    async def _fallback_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 서버 실패 시 폴백"""
        self.logger.warning(f"MCP 서버 사용 불가, 폴백 모드로 {tool_name} 실행")
        query = arguments.get("query", "")
        return {
            "query": query,
            "results": [{"title": f"폴백 결과: {query}", "content": "MCP 서버를 사용할 수 없어 제한된 결과를 제공합니다.", "source": "fallback"}],
            "total": 1,
            "fallback": True
        }

    async def cleanup(self):
        """리소스 정리"""
        for server_name, session in self.sessions.items():
            try:
                await session.close()
                self.logger.info(f"MCP 서버 '{server_name}' 종료됨")
            except Exception as e:
                self.logger.error(f"MCP 서버 '{server_name}' 종료 실패: {e}")
        self.sessions.clear()
        self.is_initialized = False
