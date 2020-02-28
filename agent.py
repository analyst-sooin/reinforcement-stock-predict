'''
* 속성
    initial_balance : 초기 투자금
    balance : 현금 잔고
    num_stock : 보유 주식 수
    portfolio_value : 포트폴리오 가치 (투자금 잔고 + 주식현재가 * 보유 주식 수

* 함수
    reset(): 에이전트의 상태를 초기화
    set_balance(): 초기 자본금을 설정
    get_states() : 에이전트 상태를 획득
    decide_action(): 탐험 또는 정책 신경망에 의한 행동 결정
    validate_action(): 행동의 유효성 판단
    decide_trading_unit(): 매수 또는 매도할 주식 수 결정
    act(): 행동 수행
'''

import numpy as np


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0  # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0  # 거래세 미고려 (실제 0.3%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2, # max_trading_unit 값을 크게 잡으면 결정한 행동에 대한 확신이 높을 때 더 많이 매수 또는 매도 할 수 있음
        delayed_reward_threshold=.05):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치
            # 손익률이 이값을 넘으면 지연 보상 발

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금생, 투작 시점의 보유 현금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
            # 즉시 보상은 핸동을 수행한 시점에서 수익이 발생한 상태면 1, 아니면 -1을 줌

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    # Agent의 속성 값을 초기화
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    # Agent의 초기 자본금을 설정
    def set_balance(self, balance):
        self.initial_balance = balance

    # Agent의 상태를 반환 함
    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    '''
    주식 보유 비율 = 보유 주식 수 / (포트폴리오 가치 / 현재주가)
    주식 보유 비율은 현재 상태에서 가장 많이 가질 수 있는 주식 수 대비 현재 보유한 주식의 비율
    이 값이 0 이면 주식을 하나도 보유하지 않은 것이고 0.5이면 최대 가질 수 있는 주식 대비 절반의 주식을 보유하고 있는것이고,
    1이면 최대로 주식을 보유하고 있는 것이다.
    ratio_hold = 0 -> 주식 보유 없음
    ratio_hold = 0.5    -> 최대의 절반 주식 보유
    ratio_hold = 1  -> 최대로 주식 보유
    주힉 수가 너무 적으면 매수의 관점에서 투자에 임하고 주식 수가 너무 많으면 매도의 관점에서 투자에 임하게 됨
    즉, 보유 주식수를 투자 행동 결정에 영향을 주기위해서 정책 신경망의 입력에 포함  
    
    포트폴리오 가치 비율 = 포트폴리오 가치 / 기준 포트폴리오 가치
    포트폴리오 가치 비율은 기준 포트폴리오 가치 대비 혀내 포트폴리오 가치의 비율
    기준 포트폴리오 가치는 직전에 목표 수익 또는 손익률을 달성했을 때의 포트폴리오 가치
    포트폴리오 가치 비율이 0에 가까우면 손실이 큰것이고 1보다 크면 수익이 발생했다는 것이다.
    ratio_portfolio_value = 0 -> 가까워 질수록 손실이 큼
    ratio_portfolio_value = 1 -> 가까워 질수록 수익이 발생   
    수익률이 목표 수익률에 가까우면 매도의 관점에서 투자를 한다. 수익률이 투자 행동 결정에 영향을 줄 수 있기 때문에
    이 값을 에이전트의 상태로 정하고 정책 신경망에 포함한다.  
    '''

    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
        else:
            exploration = False
            probs = policy_network.predict(sample)  # 각 행동(매수,매도)에 대한 확률(확률 중에서 가장 큰값을 선택하여 행동으로 결정)
            action = np.argmax(probs)
            confidence = probs[action]
        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                validity = False
        return validity

    # 정책 신경망이 결정한 행동의 확률의 높을수록 매수 또는 매도하는 단위를 크게 정해준다.
    # 높은 확률로 매수를 결정하면, 많은 주식을 매수
    # 높은 확률로 매도를 결정하면, 많은 주식을 매도
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding


    # Agent가 결정한 행동을 수행
    # action은 매수,매도를 뜻하는 0 또는 1, confidence는 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률값
    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD
        # action은 매수,매도를 뜻하는 0 또는 1, confidence는 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률값

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence) #매수 단위를 정한다
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit # 매수 후 잔금을 확인
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0: # 매수 후 잔금이 0보다 적을 수 없기 때문에 확인
                trading_unit = max(min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                ) # 결정한 매수 단위가 최대 단일 거래 단위를 넘어가면 최대 단일 거래 단위로 제한하고, 최소 거래 단위보다 최소한 1주를 매수함
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit # 매수할 단위에 수수료 적용하여 총 투자 금액 계산
            self.balance -= invest_amount  # 보유 현금을 갱신 #위에서 계산한 금액을 현재 잔금에서 빼고
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신 # 주식 보유 수를 투자 단위 만큼 늘려 준다
            self.num_buy += 1  # 매수 횟수 증가 # 통계 정보

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        # 즉시 보상 판단
        self.immediate_reward = 1 if profitloss >= 0 else -1  # 수익이 발생한 산태면 1, 그렇지 않으면 -1

        # 지연 보상 판단
        if profitloss > self.delayed_reward_threshold: # delayed_reward_threshold : 지연 보상 임계치
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
    '''
    RLTrager는 지연 보상이 0이 아닌 경우를 학습한다. 즉 지연 보상 임계치를 초과하는 수익이 났으면 이전에 했던 행동들이 잘했다고 보고 positive로 학습,
    지연 보상 임계치를 초과하는 손실이 났으면 이전 행동들에 문제가 있다고 보고 부정적으로 negative를 학습한다.  
    '''