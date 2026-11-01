import random
from config import Config
from analyzer import StrategyAnalyzer
from backtester import BacktestSimulator
import sys

class WeightOptimizer:
    def __init__(self, config: Config, train_df, test_df=None):
        self.config = config
        self.train_df = train_df
        self.test_df = test_df
        
    def _normalize(self, weights: dict) -> dict:
        """가중치의 총합이 1.0이 되도록 정규화합니다."""
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0/len(weights) for k in weights}
        return {k: v/total for k, v in weights.items()}
        
    def _create_random_individual(self, keys) -> dict:
        """무작위 유전자(가중치 비율)를 가진 새로운 개체를 생성합니다."""
        weights = {k: random.uniform(0.1, 1.0) for k in keys}
        return self._normalize(weights)
        
    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """두 부모의 유전자를 섞어(교차) 자식을 만듭니다."""
        child = {}
        for key in parent1.keys():
            # 50% 확률로 부모 중 하나의 유전자(가중치)를 물려받음
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return self._normalize(child)
        
    def _mutate(self, weights: dict, mutation_rate=0.2) -> dict:
        """일정 확률로 돌연변이를 일으켜 국소 최적화(Local Optima)에 빠지는 것을 막습니다."""
        mutated = weights.copy()
        for key in mutated.keys():
            if random.random() < mutation_rate:
                # 원래 가중치에서 -50% ~ +50% 정도 랜덤하게 변이
                change = mutated[key] * random.uniform(-0.5, 0.5)
                mutated[key] = max(0.01, mutated[key] + change)
        return self._normalize(mutated)
        
    def _evaluate_fitness(self, weights: dict, period_mode: str, market: str) -> float:
        """개체(가중치 조합)의 적합도를 평가합니다. (과거 데이터 백테스트 수익률)"""
        # 이 개체의 가중치를 임시로 config에 적용
        self.config.update_weights(market, period_mode, weights)
        
        # 전략 생성 및 시그널 산출
        analyzer = StrategyAnalyzer(self.config, market=market)
        analyzed_df = analyzer.generate_signals(self.train_df)
        
        # 백테스트로 수익률 계산
        backtester = BacktestSimulator()
        result = backtester.run(analyzed_df, period_mode=period_mode)
        
        return result['return_pct']

    def train(self, market='US', period_mode='short', epochs=10, population_size=20, learning_rate=0.2):
        """
        유전 알고리즘(Genetic Algorithm)을 사용하여 최고의 수익률을 내는 가중치를 찾습니다.
        - market: 'US' 또는 'KR'
        - epochs: 진화시킬 세대 수
        - population_size: 한 세대당 개체 수 (클수록 다양하지만 느림)
        - learning_rate: 기존 가중치 유지 비율 (0.0 ~ 1.0, 낮을수록 기존 학습 보존)
        """
        print(f"\n🧬 [유전 알고리즘] {market} {period_mode.upper()} 모드 가중치 최적화 시작 (세대: {epochs}, 인구: {population_size}, 학습률: {learning_rate})")
        
        target_weights = self.config.get_market_weights(market)
        base_weights = target_weights[period_mode].copy()
            
        keys = list(base_weights.keys())
        
        # 1. 초기 세대(Population) 생성
        population = [base_weights.copy()] # 현재 최고값을 반드시 포함 (우월 유전자 보존)
        for _ in range(population_size - 1):
            population.append(self._create_random_individual(keys))
            
        best_overall_return = -999.0
        best_overall_weights = None
            
        for generation in range(epochs):
            # 2. 적합도 평가 (Fitness Evaluation)
            scored_population = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, period_mode, market)
                scored_population.append((fitness, individual))
                
            # 수익률(Fitness) 기준으로 내림차순 정렬 (1등이 맨 위로)
            scored_population.sort(key=lambda x: x[0], reverse=True)
            
            gen_best_return = scored_population[0][0]
            gen_best_weights = scored_population[0][1]
            
            if gen_best_return > best_overall_return:
                best_overall_return = gen_best_return
                best_overall_weights = gen_best_weights.copy()
                
            weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in gen_best_weights.items()])
            print(f" - [세대 {generation+1:02d}/{epochs}] 🏆 최고 수익률: {gen_best_return:8.2f}% | 가중치: {weights_str}")
                  
            # 3. 다음 세대 생성 (Selection, Crossover, Mutation)
            new_population = []
            
            # Elitism(엘리트주의): 상위 2개 개체는 변이 없이 그대로 다음 세대로 진출 (수익률 하락 방지)
            new_population.append(scored_population[0][1])
            if population_size > 1:
                new_population.append(scored_population[1][1])
            
            # 부모 풀(Pool) 설정: 상위 50%의 우수한 개체들만 교배 자격을 얻음
            top_half = [ind for fit, ind in scored_population[:max(2, population_size//2)]]
            
            while len(new_population) < population_size:
                parent1 = random.choice(top_half)
                parent2 = random.choice(top_half)
                
                # 교차(Crossover)
                child = self._crossover(parent1, parent2)
                # 변이(Mutation)
                child = self._mutate(child, mutation_rate=0.2)
                
                new_population.append(child)
                
            population = new_population
            
        # 기존 가중치(base_weights)와 새로 찾은 최적 가중치(best_overall_weights)를 블렌딩(학습률 적용)
        # 이를 통해 과거 데이터에 대한 기억을 완전히 잃어버리는 현상(Catastrophic Forgetting)을 방지합니다.
        blended_weights = {}
        for key in keys:
            blended_weights[key] = (1 - learning_rate) * base_weights[key] + learning_rate * best_overall_weights[key]
        
        blended_weights = self._normalize(blended_weights)
            
        # 최적의 가중치로 최종 업데이트
        self.config.update_weights(market, period_mode, blended_weights)
        
        weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in blended_weights.items()])
        print(f"🎯 {market} {period_mode.upper()} 모드 최적화 완료! (학습률 {learning_rate} 반영)")
        print(f"   - 최종 반영 가중치: {weights_str}")
        return blended_weights
