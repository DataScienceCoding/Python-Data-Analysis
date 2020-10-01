
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 3),
                  index=['a', 'c', 'e', 'f',
                         'h'],
                  columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

print(df, '> df\n')

print(df['one'].isnull(), "> df['one'].isnull()\n")
print(df['one'].notnull(), "> df['one'].notnull()\n")

# 直接删除法
print(df.dropna(), "> df.dropna\n")

# 标量填充法
print(df.fillna(0), "> df.fillna(0)\n")

# 相邻前后行填充法
print(df.fillna(method='pad'), "> df.fillna(method='pad')\n")
print(df.fillna(method='backfill'), "> df.fillna(method='backfill')\n")

# 通用替换法
df = pd.DataFrame({'one': [10, 20, 30, 40, 50, 2000],
                   'two': [1000, 0, 30, 40, 50, 60]})
print(df.replace({1000: 10, 2000: 60}))



