from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class MCPServerConfig:
    """MCP 서버 설정 정보"""
    name: str
    transport: str  # "stdio" 또는 "http"
    command: Optional[str] = None  # stdio용 명령어
    args: Optional[List[str]] = None  # stdio용 인자들
    env: Optional[Dict[str, str]] = None  # 환경 변수
    url: Optional[str] = None  # http용 URL
    timeout: int = 30  # 타임아웃 (초)
    
    @classmethod
    def create_stdio_config(cls, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        """STDIO 전송용 설정 생성"""
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=args or [],
            env=env
        )
    
    @classmethod
    def create_http_config(cls, name: str, url: str, timeout: int = 30):
        """HTTP 전송용 설정 생성"""
        return cls(
            name=name,
            transport="http",
            url=url,
            timeout=timeout
        )

# 기본 MCP 서버 설정들
DEFAULT_MCP_SERVERS = {
    "web_search": MCPServerConfig.create_stdio_config(
        name="web_search",
        command="python",
        args=["./mcp_servers/web_search_server.py"]
    ),
    "api_search": MCPServerConfig.create_stdio_config(
        name="api_search", 
        command="python",
        args=["./mcp_servers/api_server.py"]
    )
}