# import matplotlib.pyplot as plt
# import numpy as np
# import numpy.ma as ma

# data = np.random.rand(25 * 25).reshape(25, -1)
# mask = np.tri(data.shape[0], k=-1)
# data_masked = ma.array(data, mask=mask)
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(data)
# ax2.imshow(data_masked)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# salary = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), 'MLB2008.csv')), delimiter=',',
#                     usecols=(1,), skiprows=1, unpack=True)

# # 创建一个数组，存放可以被3整除的数组
# triples = np.arange(0, len(salary), 3)
# print("Triples", triples[:10], "...")
# # 生成一个元素值全为1且大小与薪金数据数组相等的数组
# signs = np.ones(len(salary))
# print("Signs", signs[:10], "...")
# # 下标是3的倍数的数组元素的值取反
# signs[triples] = -1
# print("Signs", signs[:10], "...")
# # 对数组取对数
# ma_log = np.ma.log(salary * signs)
# # 打印相应的薪金数据
# print("Masked logs", ma_log[:10], "...")
# # 此处规定：所谓异常值，就是在平均值一个标准差以下或者在平均值一个标准差以上的那些数值
# dev = salary.std()
# avg = salary.mean()
# inside = np.ma.masked_outside(salary, avg - dev, avg + dev)
# print("Inside", inside[:10], "...")

# 分别绘制原始薪金数据、取对数后的数据和取幂复原后的数据以及应用基于标准差的掩码之后的数据
# plt.subplot(311)
# plt.title("Original")
# plt.plot(salary)

# plt.subplot(312)
# plt.title("Log Masked")
# plt.plot(np.exp(ma_log))

# plt.subplot(313)
# plt.title("Not Extreme")
# plt.plot(inside)

# plt.subplots_adjust(hspace=.9)
# plt.show()


# import numpy
# import scipy.misc as sm
# import matplotlib.pyplot as plt

# face = sm.face()
# # 创建一个掩码，取值非0即1
# random_mask = numpy.random.randint(0, 2, size=face.shape)

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.subplot(221)
# plt.title("原图")
# plt.imshow(face)
# plt.axis('off')

# # 创建一个掩码数组
# masked_array = numpy.ma.array(face, mask=random_mask)

# plt.subplot(222)
# plt.title("掩码处理后的原图")
# plt.imshow(masked_array)
# plt.axis('off')

# plt.subplot(223)
# plt.title("对数图")
# plt.imshow(numpy.ma.log(face).astype('float32'))
# plt.axis('off')


# plt.subplot(224)
# plt.title("掩码处理后的对数图")
# plt.imshow(numpy.ma.log(masked_array).astype("float32"))
# plt.axis('off')

# plt.show()

# import numpy as np
# from scipy.stats import shapiro
# from scipy.stats import anderson
# from scipy.stats import normaltest

# flutrends = np.loadtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), 'goog_flutrends.csv')), delimiter=',', usecols=(1, ),
#                        skiprows=1, converters={1: lambda s: float(s or 0)}, unpack=True)
# N = len(flutrends)
# normal_values = np.random.normal(size=N)

# # 返回第1个元素是一个检验统计量，第2个数值是p值
# print("Normal Values Shapiro", shapiro(normal_values))
# print("Flu Shapiro", shapiro(flutrends))

# print("Normal Values Anderson", anderson(normal_values))
# print("Flu Anderson", anderson(flutrends))

# print("Normal Values normaltest", normaltest(normal_values))
# print("Flu normaltest", normaltest(flutrends))


# import numpy as np
# import matplotlib.pyplot as plt

# N = 10000

# normal_values = np.random.normal(size=N)
# # normed参数官方不推荐使用，建议改用density参数，True为频率直方图/False为频数直方图
# # bins指定间隔数
# _, bins, _ = plt.hist(normal_values, bins=int(np.sqrt(N)), density=True, lw=1)
# # 正态分布有两个参数,即均数μ和标准差σ,可记作 N(μ, σ)
# # 均数μ决定正态曲线的中心位置；标准差σ决定正态曲线的陡峭或扁平程度.σ越小,曲线越陡峭；σ越大,曲线越扁平
# sigma = 1
# mu = 0
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2)), lw=2)
# plt.show()


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, show

score = np.zeros(10000)
score[0] = 1000
result = np.random.binomial(9, 0.5, size=len(score))

for i in range(1, len(score)):
    if result[i] < 5:
        score[i] = score[i - 1] - 1
    elif result[i] < 10:
        score[i] = score[i - 1] + 1
    else:
        raise AssertionError("Unexpected grade " + score)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("模拟10000次抛九枚硬币的试验")
plt.xlabel("试验次数")
plt.ylabel("获得分数")
plot(np.arange(len(score)), score)
show()


# A = np.mat("3 -2; 1 0")
# print("{} > 矩阵A\n".format(A))

# print("{} > 矩阵A的特征值\n".format(np.linalg.eigvals(A)))

# eigenvalues, eigenvectors = np.linalg.eig(A)
# print("{} > 矩阵A的特征值".format(eigenvalues))
# print("{} > 矩阵A的特征向量\n".format(eigenvectors))

# for i, _ in enumerate(eigenvalues):
#     print("{} > 左边".format(np.dot(A, eigenvectors[:, i])))
#     print("{} > 右边".format(eigenvalues[i] * eigenvectors[:, i]))


# 求线性方程组`Ax = b`的解
# 1. 创建矩阵A和数组b
# A = np.mat("1 -2 1; 0 2 -8; -4 5 9")
# print("{} > 矩阵A\n".format(A))

# b = np.array([0, 8, -9])
# print("{} > 数组b\n".format(b))

# # 2. 用solve()函数来解这个线性方程组
# x = np.linalg.solve(A, b)
# print('{} > 求解结果\n'.format(x))

# # 3. 用dot()函数进行验算
# print('{} > 验算结果\n'.format(np.dot(A, x)))

# 1. 用mat()函数创建一个示例矩阵
# m = np.mat('2 4 6; 4 2 6; 10 -4 18')
# print('{} > 矩阵m\n'.format(m))

# 2.矩阵求逆
# inverse = np.linalg.inv(m)
# print('{} > 矩阵m的逆\n'.format(inverse))
# 注: 如果该矩阵是奇异的，或者非方阵，那么我们就会得到LinAlgError消息
#     NumPy库中的pinv()函数可以用来求伪逆矩阵，它适用于任意矩阵，包括非方阵

# 3. 用乘法进行验算
# print("{} > 验证逆矩阵\n".format(m * inverse))

# 4. 用矩阵乘法的结果减去3×3的单位矩阵以得到求逆过程中出现的误差
# print("{} > 计算得到的逆矩阵的误差\n".format(m * inverse - np.eye(3)))

# import numpy as np
# from scipy.stats import scoreatpercentile
# import pandas as pd

# data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), 'co2.csv')), index_col=0, parse_dates=True)

# co2 = np.array(data.co2)

# print("二氧化碳最大值（方法）: ", co2.max())
# print("二氧化碳最大值（函数）: ", np.max(co2))

# print("二氧化碳最小值（方法）: ", co2.min())
# print("二氧化碳最小值（函数）: ", np.min(co2))

# print("二氧化碳平均值（方法）: ", co2.mean())
# print("二氧化碳平均值（函数）", np.mean(co2))

# print("二氧化碳标准差（方法）: ", co2.std())
# print("二氧化碳标准差（函数）: ", np.std(co2))

# print("二氧化碳中位数（median函数）: ", np.median(co2))
# print("二氧化碳中位数（scoreatpercentile函数）: ", scoreatpercentile(co2, 50))


# from pptx import Presentation
# # from pptx.enum.shapes import MSO_SHAPE
# from pptx.util import Inches, Pt
# from pptx.dml.color import RGBColor

# prs = Presentation()
# blank_slide_layout = prs.slide_layouts[6]
# slide = prs.slides.add_slide(blank_slide_layout)

# # 设置要新建的文本框的位置
# left = top = width = height = Inches(1)
# # 实例化一个文本框
# tb = slide.shapes.add_textbox(left, top, width, height)
# # 设置文件框的类型
# tf = tb.text_frame
# # 给定文本框里的文字
# tf.text = 'This is text inside a textbox'
# # 添加段落，向下在添加段落文字
# p = tf.add_paragraph()
# # 给新增加的段落添加文字
# p.text = "This is a second add_paragraph that's bold"
# # 给新添加的段落文字设置为粗体
# p.font.bold = True
# # 再在这个文本框中新建一个段落
# p = tf.add_paragraph()
# # 设置新段落的文字
# p.text = "你好重古怪哦"
# # 设置新添加的段落文字的字号为40
# p.font.name = '微软雅黑'
# p.font.size = Pt(32)
# p.font.color.rgb = RGBColor(0, 0, 0)

# prs.save("test.pptx")
