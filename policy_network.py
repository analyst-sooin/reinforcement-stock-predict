'''
* 속성
    model : lstm 신경망 모델
    prob : 가장 최근에 계싼한 투자 행동별 확률
* 함수
    reset() : prob 변수를 초기화
    predict() : 신경망을 통해 투자 행동별 확률 계산
    train_on_batch() : 배치 학습을 위한 데이터 생성
    save_model(): 학습한 신경망을 파일로 저장
    load_model(): 파일로 저장한 신경망을 로
'''


import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM 신경망
        self.model = Sequential()

        self.model.add(LSTM(256, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    def reset(self):
        self.prob = None

    #   신경망을 통해서 학습 데이터와 에이전트 상태를 합한 17차원의 입력을 받아서 매수와 매도가 수익을 높일 것으로 판단되는 확률을 구한다.
    #   predict함수는 여러 샘플을 한꺼버에 받아서 신경망의 출력을 반환한다. 하나의 샘플에 대한 결과만을 받고 싶어도 샘플의 배열로 입력값을
    #   구성해야 하기 때문에 2차원 배열로 재구성한다. np.array(sample)~
    def predict(self, sample):
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def save_model(self, model_path): # 학습한 정책 신경망을 파일로 저장함
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
