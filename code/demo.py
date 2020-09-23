import numpy as np


x = np.array([[1], [2], [3]])
print(x, '> x\n')
y = np.array([4, 5, 6])
print(y, '> y\n')

# 对 y 广播 x
b = np.broadcast(x, y)
# b 拥有 iterator 属性，基于自身组件的迭代器元组
r, c = b.iters

print(next(r), next(c), '> next(r), next(c)\n')
print(next(r), next(c), '> next(r), next(c)\n')

print(b.shape, '> b.shape\n')
# 手动使用 broadcast 将 x 与 y 相加
b = np.broadcast(x, y)
c = np.empty(b.shape)

print(c.shape, '> c.shape\n')
c.flat = [u + v for (u, v) in b]
print(c, '> c\n')
print(x + y)


a = np.arange(4).reshape(1, 4)
print(a, '> a\n')
print(np.broadcast_to(a, (4, 4)), '> np.broadcast_to(a, (4, 4))\n')

x = np.array(([1, 2], [3, 4]))
print(x, '> x\n')
print(np.expand_dims(x, axis=0), '> np.expand_dims(x, axis = 0)\n')

x = np.arange(9).reshape(1, 3, 3)
print(x, '> x\n')
y = np.squeeze(x)
print(np.squeeze(x), '> np.squeeze(x)\n')

