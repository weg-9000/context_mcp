import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# 프로젝트 경로 설정
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

try:
    from agent.RetrieverAgent.config import (
    get_all_config, validate_environment, get_llm_config
    )
    from agent.RetrieverAgent.mcp_client.client import MCPClient
    from agent.RetrieverAgent.mcp_client.config import DEFAULT_MCP_SERVERS
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("필요한 모듈이 설치되어 있는지 확인하세요.")
    sys.exit(1)

# 로깅 설정
def setup_logging(config: Dict[str, Any]):
    """로깅 설정"""
    logging_config = config.get("logging", {})
    log_level = logging_config.get("level", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class RetrieverAgentRunner:
    """RetrieverAgent 실행기"""
    
    def __init__(self):
        self.config = None
        self.mcp_client = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> bool:
        """에이전트 초기화"""
        try:
            # 환경 변수 검증
            if not validate_environment():
                return False
            
            # 설정 로드
            self.config = get_all_config()
            setup_logging(self.config)
            
            self.logger.info("RetrieverAgent 초기화 시작")
            
            # MCP 클라이언트 초기화
            self.mcp_client = MCPClient(DEFAULT_MCP_SERVERS)
            success = await self.mcp_client.initialize()
            
            if success:
                self.logger.info("MCP 클라이언트 초기화 성공")
            else:
                self.logger.warning("MCP 클라이언트 초기화 실패, 폴백 모드로 작동")
            
            # LLM 설정 확인
            llm_config = get_llm_config()
            self.logger.info(f"LLM 설정: {llm_config['provider']} - {llm_config['model']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"초기화 실패: {e}")
            return False
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """쿼리 처리"""
        if not self.mcp_client:
            return {"error": "에이전트가 초기화되지 않았습니다."}
        
        try:
            self.logger.info(f"쿼리 처리 시작: {query}")
            
            # 1. 웹 검색 수행
            web_results = await self.mcp_client.call_tool(
                "web_search", 
                "enhanced_web_search", 
                {"query": query, "max_results": 3}
            )
            
            # 2. GitHub 검색 수행
            github_results = await self.mcp_client.call_tool(
                "api_search",
                "github_search",
                {"query": query}
            )
            
            # 3. Stack Overflow 검색 수행
            stackoverflow_results = await self.mcp_client.call_tool(
                "api_search",
                "stackoverflow_search", 
                {"query": query}
            )
            
            # 결과 통합
            response = {
                "query": query,
                "web_search": web_results,
                "github_search": github_results,
                "stackoverflow_search": stackoverflow_results,
                "status": "success"
            }
            
            self.logger.info("쿼리 처리 완료")
            return response
            
        except Exception as e:
            self.logger.error(f"쿼리 처리 오류: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def run_interactive(self):
        """대화형 모드 실행"""
        print("\n" + "="*60)
        print("🤖 RetrieverAgent 대화형 모드")
        
        # 설정 정보 출력
        llm_config = get_llm_config()
        print(f"📡 LLM: {llm_config['provider']} - {llm_config['model']}")
        print(f"🔧 MCP 서버: {'활성화' if self.mcp_client.is_initialized else '폴백 모드'}")
        
        print("💬 질문을 입력하세요 (종료: 'quit', 'exit', 'q')")
        print("="*60 + "\n")
        
        while True:
            try:
                # 사용자 입력
                user_input = input("👤 질문: ").strip()
                
                # 종료 명령 확인
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 RetrieverAgent를 종료합니다.")
                    break
                
                if not user_input:
                    continue
                
                # 쿼리 처리
                print("🔍 검색 중...")
                result = await self.process_query(user_input)
                
                # 결과 출력
                if result.get("status") == "success":
                    print(f"\n🤖 검색 결과:")
                    
                    # 웹 검색 결과
                    web_results = result.get("web_search", {}).get("results", [])
                    if web_results:
                        print(f"\n📰 웹 검색 ({len(web_results)}개):")
                        for i, item in enumerate(web_results[:2], 1):
                            print(f"  {i}. {item.get('title', '제목 없음')}")
                            if item.get('url'):
                                print(f"     {item['url']}")
                    
                    # GitHub 검색 결과
                    github_results = result.get("github_search", {}).get("results", [])
                    if github_results:
                        print(f"\n💻 GitHub ({len(github_results)}개):")
                        for i, item in enumerate(github_results[:2], 1):
                            print(f"  {i}. {item.get('name', '프로젝트명 없음')}")
                            if item.get('html_url'):
                                print(f"     {item['html_url']}")
                    
                    # Stack Overflow 검색 결과
                    so_results = result.get("stackoverflow_search", {}).get("results", [])
                    if so_results:
                        print(f"\n❓ Stack Overflow ({len(so_results)}개):")
                        for i, item in enumerate(so_results[:2], 1):
                            print(f"  {i}. {item.get('title', '질문 제목 없음')}")
                            if item.get('link'):
                                print(f"     {item['link']}")
                
                else:
                    error_msg = result.get("error", "알 수 없는 오류")
                    print(f"❌ 오류: {error_msg}")
                
                print()  # 빈 줄 추가
                
            except KeyboardInterrupt:
                print("\n👋 사용자에 의해 종료되었습니다.")
                break
            except Exception as e:
                print(f"❌ 예상치 못한 오류: {e}")
    
    async def run_test(self):
        """테스트 모드 실행"""
        test_queries = [
            "Python 최신 버전",
            "LangChain 사용법", 
            "GitHub Actions 설정"
        ]
        
        print("\n" + "="*50)
        print("🧪 RetrieverAgent 테스트 모드")
        print("="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n테스트 {i}/{len(test_queries)}: {query}")
            print("-" * 30)
            
            result = await self.process_query(query)
            
            if result.get("status") == "success":
                web_count = len(result.get("web_search", {}).get("results", []))
                github_count = len(result.get("github_search", {}).get("results", []))
                so_count = len(result.get("stackoverflow_search", {}).get("results", []))
                
                print(f"✅ 성공: 웹({web_count}) GitHub({github_count}) SO({so_count})")
            else:
                print(f"❌ 실패: {result.get('error')}")
        
        print("\n🎉 모든 테스트 완료!")
    
    async def cleanup(self):
        """리소스 정리"""
        if self.mcp_client:
            await self.mcp_client.cleanup()
        self.logger.info("리소스 정리 완료")

async def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RetrieverAgent 실행")
    parser.add_argument("--mode", choices=["interactive", "test"], 
                       default="interactive", help="실행 모드")
    parser.add_argument("--query", type=str, help="단일 쿼리 실행")
    
    args = parser.parse_args()
    
    # 에이전트 실행기 생성
    runner = RetrieverAgentRunner()
    
    try:
        # 초기화
        success = await runner.initialize()
        if not success:
            print("❌ RetrieverAgent 초기화 실패")
            return 1
        
        # 실행 모드별 처리
        if args.query:
            # 단일 쿼리 모드
            print(f"🔍 쿼리 실행: {args.query}")
            result = await runner.process_query(args.query)
            
            if result.get("status") == "success":
                print("✅ 검색 완료")
                # 간단한 결과 요약 출력
                for search_type in ["web_search", "github_search", "stackoverflow_search"]:
                    if search_type in result:
                        count = len(result[search_type].get("results", []))
                        print(f"  {search_type}: {count}개 결과")
            else:
                print(f"❌ 오류: {result.get('error')}")
        
        elif args.mode == "test":
            # 테스트 모드
            await runner.run_test()
        
        else:
            # 대화형 모드 (기본)
            await runner.run_interactive()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        return 0
    except Exception as e:
        print(f"실행 중 오류: {e}")
        return 1
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"치명적 오류: {e}")
        sys.exit(1)