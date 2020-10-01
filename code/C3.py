import pandas as pd
import numpy as np
from datetime import datetime
import quandl
import os

# 创建一个空的系列
s = pd.Series()
print(s, '\n')

# 从ndarray创建一个系列
data = np.array(['a', 'b', 'c', 'd'])
s = pd.Series(data)
print(s, '\n')

data = np.array(['a', 'b', 'c', 'd'])
s = pd.Series(data, index=[100, 101, 102, 103])
print(s, '\n')

print('------')

# 从字典创建一个系列
data = {'a': 0., 'b': 1., 'c': 2.}
s = pd.Series(data)
print(s, '\n')

data = {'a': 0., 'b': 1., 'c': 2.}
s = pd.Series(data, index=['b', 'c', 'd', 'a'])
print(s, '\n')

print('------')

# 从标量创建一个系列
s = pd.Series(5, index=[0, 1, 2, 3])
print(s, '\n')

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s[0], '\n')

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s[:3], '\n')

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s[-3:], '\n')

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s['a'], '\n')

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s[['a', 'c', 'd']], '\n')

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s['f'], '\n')

print('# 创建一个空的数据框')
df = pd.DataFrame()
print(df, '\n')

print('# 从列表创建数据框')
data = [1, 2, 3, 4, 5]
df = pd.DataFrame(data)
print(df, '\n')

data = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
df = pd.DataFrame(data, columns=['Name', 'Age'])
print(df, '\n')

data = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
df = pd.DataFrame(data, columns=['Name', 'Age'], dtype=float)
print(df, '\n')

print('# 从ndarrays/Lists的字典来创建数据框')
data = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
df = pd.DataFrame(data)
print(df, '\n')

print('# 使用数组创建一个索引的数据框')
data = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
df = pd.DataFrame(data, index=['rank1', 'rank2', 'rank3', 'rank4'])
print(df, '\n')

print('# 从列表创建数据框')
data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print(df, '\n')

data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
print(df, '\n')

data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])
df2 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1'])
print(df1, '> df1\n')
print(df2, '> df2\n')

print('# 从系列的字典来创建数据框')
d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df, '\n')

print('# 从数据框中选择一列')
d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df['one'])

print('# 向现有数据框添加一个新列')
d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df['three'] = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(df, '\n')
df['four'] = df['one'] + df['three']
print(df, '\n')

print('# 列删除与弹出')
d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']),
     'three': pd.Series([10, 20, 30], index=['a', 'b', 'c'])}
df = pd.DataFrame(d)
print(df, '\n')
del df['one']
print(df, '\n')
df.pop('two')
print(df, '\n')

print('# 行选择—标签选择')
d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df.loc['b'])

print('# 行选择—按整数位置选择')
d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df.iloc[2])

print('# 行切片 (使用冒号运算符选择多行)')
d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df[2:4])

print('# 行切片—附加行')
df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])
df = df.append(df2)
print(df, '\n')

print('# 行切片—删除行')
df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])
df = df.append(df2)
df = df.drop(0)
print(df, '\n')

s = pd.Series(np.random.randn(4))
print(s, '> s\n')

# 返回系列的标签列表
print(s.axes, '> s.axes\n')

# empty示例
print(s.empty, '> s.empty\n')

# ndim示例
print(s.ndim, '> s.ndim\n')

# size示例
print(s.size, '> s.size\n')

# values示例
print(s.values, '> s.values\n')

# head()和tail()方法示例
print(s.head(2), '> s.head(2)\n')
print(s.tail(2), '> s.tail(2)\n')


quandl.ApiConfig.api_key = "AFJ7nyGtmQQkijbz1s2n"
sunspots = quandl.get('SIDC/SUNSPOTS_A')
print(type(sunspots), '> type of sunspots\n')
print('sunspots.dtypes', sunspots.dtypes, '\n')
print('Describe', sunspots.describe(), '\n')
print('Kurtosis', sunspots.kurt(), '\n')
print(sunspots['Number of Observations'].corr(
    sunspots['Yearly Mean Total Sunspot Number']), '\n')

print('当前程序执行目录: ', os.getcwd())
df = pd.read_csv("code/temp.csv")
print(df, '\n')

print('# 自定义索引')
df = pd.read_csv("code/temp.csv", index_col=['S.No'])
print(df, '\n')

print('# 转换器')
df = pd.read_csv("code/temp.csv", dtype={'Salary': np.float64})
print(df.dtypes, '\n')

print('# 使用names参数指定标题(同时需要使用header参数来删除原有标题)')
df = pd.read_csv("code/temp.csv", names=['a', 'b', 'c', 'd', 'e'], header=0)
print(df, '\n')

print('# 跳过指定行')
df = pd.read_csv("code/temp.csv", skiprows=2)
print(df, '\n')


left = pd.DataFrame({
    'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
    'subject_id': ['sub1', 'sub2', 'sub4', 'sub6', 'sub5'],
    'Marks_scored': [98, 90, 87, 69, 78]},
    index=[1, 2, 3, 4, 5])

right = pd.DataFrame({
    'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
    'subject_id': ['sub2', 'sub4', 'sub3', 'sub6', 'sub5'],
    'Marks_scored': [89, 80, 79, 97, 88]},
    index=[1, 2, 3, 4, 5])

print(pd.concat([left, right]), '> pd.concat([left, right])\n')
print(pd.concat([left, right], keys=['x', 'y'], ignore_index=True),
      "> pd.concat([left, right], keys=['x', 'y'], ignore_index=True)\n")
print(pd.concat([left, right], axis=1), '> pd.concat([left, right], axis=1)\n')
print(pd.merge(left, right, on='subject_id'),
      "> pd.merge(left, right, on='subject_id')\n")
print(pd.merge(left, right, on=['subject_id'], how='left'),
      "> pd.merge(left, right, on=['subject_id'], how='left')\n")

ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
                     'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
            'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
            'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
            'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
df = pd.DataFrame(ipl_data)
grp = df.groupby('Team')
print(grp, '> group\n')
# 查看分组
print(grp.groups, '> grp.groups\n')
# 遍历分组
for name, group in grp:
    print(name, group, '\n')

# 选择分组
print(grp.get_group('Riders'), "> grp.get_group('Riders')\n")

# 聚合
print(grp['Points'])
print(grp['Points'].agg([np.mean, np.sum]),
      "> grp['Kings'].agg(np.mean, np.sum)\n")
print(grp['Points'].agg(np.size), "> grp['Kings'].agg(np.size)\n")
# 转换
print(grp.transform(lambda x: (x - x.mean()) / x.std()*10),
      "> grp.transform(lambda x: (x - x.mean()) / x.std()*10)\n")
print(grp.filter(lambda x: len(x) >= 3),
      "> grp.filter(lambda x: len(x) >= 3)\n")


print('获取当前时间')
print(datetime.now(), '\n')

print('根据指定格式创建日期')
print(datetime(2017, 11, 5), "> pd.datetime(2017, 11, 5)\n")
print(pd.Timestamp('2018-11-01'), ">  pd.Timestamp('2018-11-01')\n")
print(pd.Timestamp(1588686880, unit='s'),
      "> pd.Timestamp(1588686880, unit='s')\n")

print('创建一个时间范围')
print(pd.date_range('12:00', '23:59', freq='30min').time,
      "> pd.date_range('12:00', '23:59', freq='30min').time\n")
print(pd.date_range('12:00', '23:59', freq='H').time,
      "> pd.date_range('12:00', '23:59', freq='H').time\n")
print(pd.bdate_range('2011/11/03', periods=5),
      "> pd.bdate_range('2011/11/03', periods=5)\n")

print('日期转换')
print(pd.to_datetime(pd.Series(['2020-10-01', '2019-10-10']), format='%Y-%m-%d'),
      "> pd.to_datetime(pd.Series(['Jul 31, 2009','2019-10-10', None])\n")
print(pd.to_datetime(['2009/11/23', '2019.12.31', None]),
      "> pd.to_datetime(['2009/11/23', '2019.12.31', None])\n")

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