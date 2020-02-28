# from pandas_datareader import data
# chart_data = data.DataReader("KRX:005930", "google")
#
#
# """
# <html><head><meta http-equiv="content-type" content="text/html; charset=utf-8"/><title>Sorry...</title><style> body { font-family: verdana, arial, sans-serif; background-color: #fff; color: #000; }</style></head><body><div><table><tr><td><b><font face=sans-serif size=10><font color=#4285f4>G</font><font color=#ea4335>o</font><font color=#fbbc05>o</font><font color=#4285f4>g</font><font color=#34a853>l</font><font color=#ea4335>e</font></font></b></td><td style="text-align: left; vertical-align: bottom; padding-bottom: 15px; width: 50%"><div style="border-bottom: 1px solid #dfdfdf;">Sorry...</div></td></tr></table></div><div style="margin-left: 4em;"><h1>We\'re sorry...</h1><p>... but your computer or network may be sending automated queries. To protect our users, we can\'t process your request right now.</p></div><div style="margin-left: 4em;">See <a href="https://support.google.com/websearch/answer/86640">Google Help</a> for more information.<br/><br/></div><div style="text-align: center; border-top: 1px solid #dfdfdf;"><a href="https://www.google.com">Google Home</a></div></body></html>
# """
#
#
# from pandas_datareader import data
# import fix_yahoo_finance
# fix_yahoo_finance.pdr_override()
#
# chart_data = data.get_data_yahoo(
#     '005930.KS',  # 코스피: KS, 코스닥: KQ
#     '2016-01-01',
#     '2017-12-31'
# )
import pandas as pd
import pandas_datareader as pdr
# 1번 방법
# DataReader API를 통해서 yahoo finance의 주식 종목 데이터를 가져온다.
df1 = pdr.DataReader('068270.KS', 'yahoo')
# 2번 방법
# get_data_yahoo API를 통해서 yahho finance의 주식 종목 데이터를 가져온다.
df2 = pdr.get_data_yahoo('068270.KS')
# 선택적으로 일정 기간동안의 주식 정보를 가져오는 방법입니다.
from datetime import datetime
start = datetime(2017,1,1)
end = datetime(2020,2,14)
df1 = pdr.DataReader('068270.KS', 'yahoo', start, end)
df2 = pdr.get_data_yahoo('068270.KS', start, end)

print(df1)
df1 = pd.DataFrame(df1,columns=['Open','High','Low','Close','Volume'])
print(df1)
df1.to_csv('kosp_test.csv',header=False,index=True)