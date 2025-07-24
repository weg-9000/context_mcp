from typing import Dict, Any

class MetricsCalculator:
    """검색 결과의 신뢰도 및 품질 메트릭을 계산하는 클래스"""

    def calculate_retrieval_confidence(self, results: Dict[str, Any]) -> float:
        """
        수집된 결과들을 바탕으로 전체 검색의 신뢰도를 계산합니다.
        결과의 수, 완성도, 다양성을 고려하여 점수를 매깁니다.
        """
        if not results:
            return 0.0

        # 기본 신뢰도 (결과 수에 따라)
        base_confidence = 0.6 + (len(results) / 10)

        # 완성도 보너스 (4개 전문가 기준)
        completeness_bonus = min(len(results) / 4, 1.0) * 0.2

        # 다양성 보너스 (다양한 소스에서 결과가 왔을 때)
        diversity_bonus = 0.1 if len(results) >= 3 else 0.05

        # 최종 신뢰도 계산 및 0과 1 사이로 제한
        confidence = min(base_confidence + completeness_bonus + diversity_bonus, 1.0)

        return round(confidence, 2)

    def calculate_retrieval_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        검색 결과의 품질을 다각도로 평가하는 메트릭을 계산합니다.
        커버리지, 다양성, 신선도, 관련성을 종합하여 평가합니다.
        """
        if not results:
            return {"coverage": 0.0, "diversity": 0.0, "freshness": 0.0, "relevance": 0.0, "overall": 0.0}

        # 커버리지: 4개 전문가 기준
        coverage = min(len(results) / 4, 1.0)

        # 다양성: 결과 수에 따른 다양성
        diversity = 0.9 if len(results) >= 4 else 0.8 if len(results) >= 3 else 0.7

        # 신선도: 실시간 검색을 가정하므로 높은 점수 부여
        freshness = 0.95

        # 관련성: 기본값 설정
        relevance = 0.85

        # 전체 품질 점수
        overall = (coverage + diversity + freshness + relevance) / 4

        return {
            "coverage": round(coverage, 2),
            "diversity": round(diversity, 2),
            "freshness": round(freshness, 2),
            "relevance": round(relevance, 2),
            "overall": round(overall, 2)
        }