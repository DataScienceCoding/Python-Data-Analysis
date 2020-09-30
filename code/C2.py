import numpy as np

a = np.array([1, 2, 3])
print(a)

a = np.array([[1, 2], [3, 4]])
print(a)

a = np.array([1, 2, 3, 4, 5], ndmin=2)
print(a)

a = np.array([1, 2, 3], dtype=complex)
print(a)

# print(np.sctypeDict.keys())

# 只有一个维度
a = np.arange(24)
print(a.ndim)

# 拥有三个维度
b = a.reshape(2, 4, 3)
print(b.ndim)
print(b)

# 数组的 dtype 为 int8 (占一个字节空间)
x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
print(x.itemsize)

# 数组的 dtype 现在为 float64 (占八个字节空间)
y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print(y.itemsize)

x = np.empty([3, 2], dtype=int)
print(x)

x = np.zeros([3, 2], dtype=int)
print(x)

x = (1, 2, 3) 
a = np.asarray(x, dtype=float)
print(a)

ls = range(5)
it = iter(ls)
x = np.fromiter(it, dtype=float)
print(x)

x = np.arange(10, 20, 2)
print(x)

a = np.arange(10)
s = slice(2, 7, 2)
print(a, s, a[s])

a = np.arange(10)
b = a[2:7:2]
print(b)

a = np.arange(10)
b = a[5]
print(a, b)

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a)
print('---------')
print(a[1:])
print('=========')

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a)
print('-------')
# 第2列元素
print(a[..., 1])
print('-------')
# 第2行元素
print(a[1, ...])
print('-------')
# 第2列及剩下的所有元素
print(a[..., 1:])

x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0, 1, 2], [0, 1, 0]]
print(y, '\n')

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print(x, ':x\n')
rows = np.array([[0, 0], [3, 3]])
print(rows, ':rows\n')
cols = np.array([[0, 2], [0, 2]])
print(cols, ':cols\n')
y1 = x[rows, cols]
print(y1, ':y1\n')
print(x[[0, 0, 3, 3], [0, 2, 0, 2]], ':y2')

print('-------')

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a, ':a\n')
b = a[1:3, 1:3]
print(b, ':b\n')
c = a[1:3, [1, 2]]
print(c, ':c\n')
d = a[..., 1:]
print(d, ':d\n')

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print(x, ':x\n')
print(x[x > 5], ':x>5\n')

a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
print(a[~np.isnan(a)], '\n')

x = np.arange(32).reshape((8, 4))
print(x, ':x\n')
print(x[[4, 2, 1, 7]], ':x[[4, 2, 1, 7]]\n')
print(x[[-4, -2, -1, -7]], ':x[[-4, -2, -1, -7]]\n')
print(
  x[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])],
  ':x[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]\n'
)

print('-------')

a = np.array([
  [0, 0, 0],
  [10, 10, 10],
  [20, 20, 20],
  [30, 30, 30]
])
print(a, ':a\n')
b = np.array([1, 2, 3])
print(b, ':b\n')
print(a + b, ':a+b\n')
bb = np.tile(b, (4, 1))
print(a + bb, ':a+bb\n')

print('-------')

a = np.arange(8)
print(a, ':a\n')

b = a.reshape(4, 2, order='F')
print(b, ':b\n')

a = np.arange(9).reshape(3, 3)
print(a, ':a\n')
for row in a:
    print(row)
print()
for el in a.flat:
    print(el)
print()
a = np.arange(8).reshape(2, 4)
print(a, ':a\n')

print(a.flatten(), ':a.flatten\n')
print(a.flatten(order='F'), ":a.flatten(order='F')\n")

print(a.ravel(), ':a.ravel\n')
print(a.ravel(order='F'), ":a.ravel(order='F')\n")

print('-------')

a = np.arange(12).reshape(3, 4)
print(a, ':a\n')
print(np.transpose(a), ':np.transpose(a)\n')
print(a.T, ':a.T\n')

a = np.arange(8).reshape(2, 2, 2)
print(a, ':a\n')
print(np.where(a == 6), ':np.where(a==6)\n')
print(a[1, 1, 0], ':a[1, 1, 0]\n')
print(np.rollaxis(a, 2, 0), ':np.rollaxis(a, 2, 0)\n')
print(np.where(b == 6))
print(np.swapaxes(a, 2, 0), ':np.swapaxes(a, 2, 0)\n')

x = np.array([[1], [2], [3]])
print(x, ':x\n')
y = np.array([4, 5, 6])
print(y, ':y\n')

# 对 y 广播 x
b = np.broadcast(x, y)
# b拥有 iterator 属性，基于自身组件的迭代器元组
r, c = b.iters
print(next(r), next(c), ':next(r), next(c)\n')
print(next(r), next(c), ':next(r), next(c)\n')

print(b.shape, ':b.shape\n')
# 手动使用 broadcast 将 x 与 y 相加
b = np.broadcast(x, y)
c = np.empty(b.shape)

print(c.shape, ':c.shape\n')
c.flat = [u + v for (u, v) in b]

print(c, ':c\n')
print(x + y, ':x+y\n')


a = np.array([[1, 2], [3, 4]])
print(a, '> a\n')
b = np.array([[5, 6], [7, 8]])
print(b, '> b\n')

# 沿轴 0 连接两个数组
print(np.concatenate((a, b)), '> concatenate((a, b)\n')
# 沿轴 1 连接两个数组
print(np.concatenate((a, b), axis=1), '> concatenate((a, b), axis=1)\n')

# 沿轴 0 堆叠两个数组
print(np.stack((a, b), 0), '> stack((a, b), 0)\n')
# 沿轴 1 堆叠两个数组
print(np.stack((a, b), 1), '> stack((a, b), 1)\n')
# 水平堆叠两个数组
print(np.hstack((a, b)), '> hstack((a, b))\n')
# 垂直堆叠两个数组
print(np.vstack((a, b)), '> vstack((a, b))\n')


a = np.arange(9)
print(a, '> a\n')

# 将数组a拆分为三个大小相等的子数组
print(np.split(a, 3), '> split(a, 3)\n')
# 将数组按一维数组中给定的索引位置分割
print(np.split(a, [4, 7]), '> .split(a, [4, 7])\n')

a = np.floor(10 * np.random.random((2, 6)))
print(a, '> a\n')

print(np.hsplit(a, 3), '> hsplit(a, 3)\n')

a = np.arange(16).reshape(4, 4)
print(a, '> a\n')

print(np.vsplit(a, 2), '> vsplit(a, 2)\n')