
import pandas as pd


print('通过字符串创建时间差')
print(pd.Timedelta('2 days 2 hours 15 minutes 30 seconds'), '\n')

print('通过整数创建时间差')
print(pd.Timedelta(6, unit='h'), '\n')
print(pd.Timedelta(days=2), '\n')

print('相加操作')
s = pd.Series(pd.date_range('2018-1-1', periods=3, freq='D'))
td = pd.Series([pd.Timedelta(days=i) for i in range(3)])
df = pd.DataFrame(dict(A=s, B=td))
df['C'] = df['A'] + df['B']
df['D'] = df['C'] - df['B']
print(df, '\n')
