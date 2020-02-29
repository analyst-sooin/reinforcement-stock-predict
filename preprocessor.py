import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')


def calc_mdd(list_x, list_pv):
    """
    MDD(Maximum Draw-Down) 계산
    :param list_pv: 포트폴리오 가치 리스트
    :return:
    """
    list_x = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in list_x]
    arr_pv = np.array(list_pv)
    peak_lower = np.argmax(np.maximum.accumulate(arr_pv) - arr_pv)
    peak_upper = np.argmax(arr_pv[:peak_lower])

    idx_min = np.argmin(arr_pv)
    idx_max = np.argmax(arr_pv)

    # ------------------------------
    fig, ax = plt.subplots()
    ax.plot(list_x, arr_pv, color='gray')
    #     ax.plot([list_x[peak_upper]], [arr_pv[peak_upper]], '>', color='red')
    #     ax.plot([list_x[peak_lower]], [arr_pv[peak_lower]], '<', color='blue')
    ax.plot([list_x[peak_upper], list_x[peak_lower]], [arr_pv[peak_upper], arr_pv[peak_lower]], '-', color='blue')
    ax.plot([list_x[idx_min]], [arr_pv[idx_min]], 'v', color='blue')
    ax.plot([list_x[idx_max]], [arr_pv[idx_max]], '^', color='red')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    plt.show()
    # ------------------------------

    return (arr_pv[peak_lower] - arr_pv[peak_upper]) / arr_pv[peak_upper]


import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.getcwd()))
import ensembley.RLTrader.data_manager as data_manager

# code = '005930'  # 삼성전자
# code = '000660'  # SK하이닉스
# code = '005380'  # 현대차
# code = '051910'  # LG화학
# code = '035420'  # NAVER
code = '030200'  # KT


chart_data = data_manager.load_chart_data('chart_data/{}.csv'.format(code))
chart_data = chart_data[chart_data['date'] <= '2016-12-31']
chart_data.tail()


_chart_data = chart_data[(chart_data['date'] >= '2016-01-01') & (chart_data['date'] <= '2016-12-31')]
print(_chart_data['close'].max())
print(_chart_data['close'].min())
print(calc_mdd(_chart_data['date'], _chart_data['close']))

chart_data.tail()

preprocessed_chart_data = data_manager.preprocess(chart_data)
preprocessed_chart_data[['close_ma5', 'volume_ma5', 'close_ma10', 'volume_ma10', 'close_ma20', 'volume_ma20', 'close_ma60', 'volume_ma60', 'close_ma120', 'volume_ma120']].tail()

pd.set_option('display.max_columns', 500)
training_data = data_manager.build_training_data(preprocessed_chart_data)
training_data[['open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio', 'close_lastclose_ratio', 'volume_lastvolume_ratio']].tail()

training_data[['close_ma5_ratio', 'volume_ma5_ratio', 'close_ma10_ratio', 'volume_ma10_ratio', 'close_ma20_ratio', 'volume_ma20_ratio']].tail()

training_data[['close_ma60_ratio', 'volume_ma60_ratio', 'close_ma120_ratio', 'volume_ma120_ratio']].tail()

len(training_data[(training_data['date'] >= '2017-01-01') & (training_data['date'] <= '2017-12-31')])