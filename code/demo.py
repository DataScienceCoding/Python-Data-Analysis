import numpy as np

# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a, '> a\n')
 
# print(a.shape, '> a.shape\n')
# b = np.resize(a, (3, 2))
# print(b, '> b\n')
# print(b.shape, '> b.shape\n')
# print(np.resize(a, (3, 3)), '> resize(a, (3, 3))\n')

# # 向数组添加元素
# print(np.append(a, [7, 8, 9]), '> append(a, [7, 8, 9])\n')
# # 沿轴 0 添加元素
# print(np.append(a, [[7, 8, 9]], axis=0), '> append(a, [[7, 8, 9]], axis=0)\n')
# # 沿轴 1 添加元素
# print(np.append(a, [[5, 5, 5], [7, 8, 9]], axis=1), '> append(a, [[5, 5, 5], [7, 8, 9]], axis=1)\n')

# # [无 Axis 参数] 在插入之前输入数组会被展开
# print(np.insert(a, 3, [11, 12]), '> insert(a, 3, [11, 12])\n')
# # [有 Axis 参数] 以广播方式处理输入数组
# # 沿轴 0 广播
# print(np.insert(a, 1, [11], axis=0), '> insert(a, 1, [11], axis=0)\n') 
# # 沿轴 1 广播
# print(np.insert(a, 1, 11, axis=1), '> insert(a, 1, 11, axis=1)\n')

# a = np.arange(12).reshape(3, 4)
# print(a, '> a\n')

# # [无 Axis 参数] 在插入之前输入数组会被展开
# print(np.delete(a, 5), '> delete(a, 5)\n')
# # 删除第二列
# print(np.delete(a, 1, axis=1), '> delete(a, 1, axis=1)\n')
# # 包含从数组中删除的替代值的切片
# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(np.delete(a, np.s_[::2]), '> delete(a, np.s_[::2])\n')


a = np.array([5, 2, 6, 2, 7, 5, 6, 8, 2, 9])
print(a, '> a\n')

u = np.unique(a)
print(u, '> u\n')
 
# 返回去重后的索引（数组）
u, indices = np.unique(a, return_index=True)
print(indices, '> indices\n')
 
print(a, '> a\n')

# 返回去重数组的下标
u, indices = np.unique(a, return_inverse=True)
print(u, '> u\n', indices, '> indices\n')
  
print(u[indices], '> u[indices]\n')
 
# 返回去重元素的重复数量
u, indices = np.unique(a, return_counts=True)
print(u, '> u\n', indices, '> indices\n')
