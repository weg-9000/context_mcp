"""MCP 서버 설정 관리"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class MCPServerConfig:
    """MCP 서버 설정 정보"""
    name: str
    transport: str  # "stdio" 또는 "http"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    timeout: int = 30

    @classmethod
    def create_stdio_config(cls, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        """STDIO 전송용 설정 생성"""
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=args or [],
            env=env or {}
        )

def get_mcp_servers_config() -> Dict[str, MCPServerConfig]:
    """환경 변수 기반 MCP 서버 설정"""
    
    # 현재 디렉토리 기준으로 mcp_servers 경로 설정
    current_dir = Path(__file__).parent
    mcp_servers_dir = current_dir.parent / "mcp_servers"
    
    # 환경 변수 설정
    base_env = {
        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", ""),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
    }
    
    return {
        "web_search": MCPServerConfig.create_stdio_config(
            name="web_search",
            command="python",
            args=[str(mcp_servers_dir / "web_search_server.py")],
            env=base_env
        ),
        "api_search": MCPServerConfig.create_stdio_config(
            name="api_search",
            command="python",
            args=[str(mcp_servers_dir / "api_server.py")],
            env=base_env
        )
    }

# 기본 MCP 서버 설정
DEFAULT_MCP_SERVERS = get_mcp_servers_config()