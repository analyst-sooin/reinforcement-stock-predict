'''
* 속성
    chart_data : 주식 종목의 차트 데이터
    observation : 현재 관측치
    idx : 차트 데이터에서의 현재 위치

* 함수
    reset(): idx와 observation을 초기화
    observe() : idx를 다음 위치로 이동하고 observation을 업데이트
    get_price() :  현재 observation에서 종가를 획득
'''

class Environment:

    PRICE_IDX = 4 # 종가의 위치

    def __init__(self,chart_data=None):
        self.chart_data = chart_data
        self.observation = None # 현재 위치의 관측 값
        self.idx = -1 # 현재위치를 저장

    def reset(self): #차트 데이터의 처음으로 돌아가게 함 -> idx와 observation을 초기화
        self.observation = None
        self.idx = -1

    def observe(self): #하루 앞으로 이동하며 차트데이터에서 관측 데이터 observation을 제공
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data