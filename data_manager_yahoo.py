import pandas as pd
import sqlite3
import pandas_datareader.data as web
import datetime
import timedelta



def code_ks(code_txt=pd.read_csv('kost_kost', sep=',', encoding='UTF-8')):
    code_1 = code_txt['0']
    code = code_1.tolist()
    return code



def code_yahoo(code):
    start = datetime.datetime(2015,1,1)
    end = datetime.datetime.today()
    # start = end - timedelta
    columnList = ['High', 'Low', 'Open', 'Close', 'Volume']
    stock = pd.DataFrame(columns=columnList)
    for c in code_ks():
        df = web.DataReader(c, 'yahoo', start, end)
        df['Code'] = c
        stock = pd.concat([stock, df])
        stock = stock.filter(columnList)
    stock.to_csv('kosp_test.csv',header=False, index=True)


if __name__ == '__main__':
    code_yahoo(code_ks())

