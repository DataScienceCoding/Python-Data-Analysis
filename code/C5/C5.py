import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = "Microsoft YaHei"  # 设置支持中文字体
plt.rcParams['axes.facecolor'] = '#E5E5E5'  # 设置背景色

plt.subplot(111)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y1 = np.array([866, 2335, 5710, 6482, 6120, 1605, 3813, 4428, 4631])
y2 = np.array([0.544, 0.324, 0.390, 0.411, 0.321, 0.332, 0.922, 0.029, 0.157])

plt.plot(x, y1, color='k', linestyle='-', linewidth=1,
         marker='o', markersize=3, label='注册人数')
plt.xlabel('月份')
plt.ylabel('注册量')
plt.legend(loc='upper left')
plt.twinx()  # 调用twinx方法
plt.plot(x, y2, color='r', linestyle='-.', linewidth=1,
         marker='o', markersize=3, label='激活率')
plt.xlabel('月份')
plt.ylabel('激活率')
plt.legend()
plt.title('XXX公司1-9月注册量与激活率')
plt.show()

# plt.subplot(111)
# # 指明x和y的值
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# y1 = np.array([866, 2335, 5710, 6482, 6120, 1605, 3813, 4428, 4631])
# y2 = np.array([433, 1167, 2855, 3241, 3060, 802, 1906, 2214, 2315])

# # 直接绘制折线图和柱形图
# plt.plot(x, y1, linestyle='-', color='k', linewidth=1,
#          marker='o', markersize=3, label='注册人数')
# plt.bar(x, y2, color='k', label='激活人数')

# # 设置标题
# plt.title('XXX公司1-9月注册与激活人数', loc='center')
# # 数据标签
# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11)
# for a, b in zip(x, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11)
# plt.xlabel('月份')
# plt.ylabel('注册量')
# # 设置x轴和y轴的刻度
# plt.xticks(np.arange(1, 10, 1), [
#            "1月份", "2月份", "3月份", "4月份", "5月份", "6月份", "7月份", "8月份", "9月份"])
# plt.yticks(np.arange(1000, 7000, 1000), [
#            "1000人", "2000人", "3000人", "4000人", "5000人", "6000人"])
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 6))
# # 指明x和y的值
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# y1 = np.array([866, 2335, 5710, 6482, 6120, 1605, 3813, 4428, 4631])
# y2 = np.array([433, 1167, 2855, 3241, 3060, 802, 1906, 2214, 2315])
# # 直接绘制两条折线
# plt.plot(x, y1, color='k', linestyle='-', linewidth=1,
#          marker='o', markersize=4, label='注册人数')
# plt.plot(x, y2, color='r', linestyle='--', linewidth=1,
#          marker='o', markersize=4, label='激活人数')
# # 设置标题
# plt.title('XXX公司1-9月注册与激活人数', loc='center')
# # 添加数据标签
# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11)
# for a, b in zip(x, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11)

# # 设置x轴和y轴的名称
# plt.xlabel('月份')
# plt.ylabel('注册量')
# # 设置x轴和y轴的刻度
# plt.xticks(np.arange(1, 10, 1), [
#            "1月份", "2月份", "3月份", "4月份", "5月份", "6月份", "7月份", "8月份", "9月份"])
# plt.yticks(np.arange(1000, 7000, 1000), [
#            "1000人", "2000人", "3000人", "4000人", "5000人", "6000人"])
# # 设置图例
# plt.legend()
# plt.show()

# plt.subplot(1, 2, 1)
# plt.axhline(y=2, xmin=0.2, xmax=0.6)
# plt.subplot(1, 2, 2)
# plt.axvline(x=2, ymin=0.2, ymax=0.6)
# plt.show()


# # 指定每一块的大小
# size = np.array([3.4, 0.693, 0.585, 0.570, 0.562, 0.531,
#                  0.530, 0.524, 0.501, 0.478, 0.468, 0.436])
# # 指定每一块标签文字
# xingzuo = np.array(["未知", "摩羯座", "天秤座", "双鱼座", "天蝎座", "金牛座",
#                     "处女座", "双子座", "射手座", "狮子座", "水瓶座", "白羊座"])
# # 指定每一块数值标签
# rate = np.array(["34%", "6.93%", "5.85%", "5.70%", "5.62%", "5.31%", "5.30%",
#                  "5.24%", "5.01%", "4.78%", "4.68%", "4.36%"])
# # 指定每一块的颜色
# colors = ["steelblue", "#9999ff", "red",
#           "indianred", "green", "yellow", "orange"]
# # 绘图
# plot = squarify.plot(sizes=size, label=xingzuo, color=colors,
#                      value=rate, edgecolor='white', linewidth=3)
# # 设置标题
# plt.title("菊粉星座分布", fontdict={'fontsize': 12})
# # 去除坐标轴
# plt.axis('off')
# # 去除上边框和右边框的刻度
# plt.tick_params(top=False, right=False)
# plt.show()


# plt.subplot(1, 1, 1, polar=True)  # 参数polar等于True表示建立一个极坐标系
# DataLenth = 5  # 把整个圆均分成5份
# # np.linspace 表示在指定的间隔内返回均匀间隔的数字
# angles = np.linspace(0, 2*np.pi, DataLenth, endpoint=False)
# labels = ['沟通能力', '业务理解能力', '逻辑思维能力', '快速学习能力', '工具使用能力', '其他']
# data = [2, 3, 4, 5, 6]
# data = np.concatenate((data, [data[0]]))  # 闭合
# angles = np.concatenate((angles, [angles[0]]))  # 闭合

# print(len(angles), len(data), len(labels))
# plt.polar(angles, data, color='r', marker='o')  # 绘图
# plt.xticks(angles, labels)  # 设置x轴刻度
# plt.title('某数据分析师的综合评级')
# plt.show()

# # 环形图，建立坐标系
# plt.subplot(1, 1, 1)

# # 指明x值
# x1 = np.array([8566, 5335, 7310, 6482])
# x2 = np.array([4283, 2667, 3655, 3241])

# # 绘图
# labels = ["东区", "北区", "南区", "西区"]
# plt.pie(x1, labels=labels, radius=1.0, wedgeprops={
#         "width": 0.3, "edgecolor": "w"})
# plt.pie(x2, radius=0.7, wedgeprops={"width": 0.3, "edgecolor": "w"})

# # 添加注释
# plt.annotate("完成量", xy=(0.35, 0.35), xytext=(0.7, 0.45),
#              arrowprops={"facecolor": "black", "arrowstyle": "->"})
# plt.annotate("任务量", xy=(0.75, 0.2), xytext=(1.1, 0.2),
#              arrowprops={"facecolor": "black", "arrowstyle": "->"})

# plt.title("全国各区域任务量与完成量占比", loc="center")
# plt.show()

# plt.subplot(1, 1, 1)
# x = np.array([8566, 5335, 7310, 6482])
# labels = ["东区", "北区", "南区", "西区"]
# explode = [0.1, 0, 0, 0]
# labeldistance = 1.1
# plt.pie(x, labels=labels, explode=explode, autopct='%.0f%%',
#         shadow=True, radius=1.0, labeldistance=labeldistance)

# plt.title("全国个区域任务占比", loc="center")
# plt.show()

# y1 = np.array([866, 2335, 5710, 6482, 6120, 1605, 3813, 4428, 4631])
# y2 = np.array([433, 1167, 2855, 3241, 3060, 802, 1906, 2214, 2315])
# x = [y1, y2]

# # 绘图
# labels = ["注册人数", "激活人数"]
# plt.boxplot(x, labels=labels, vert=True, widths=[0.2, 0.5])
# plt.title("XXX公司1-9月注册与激活人数", loc="center")
# plt.grid(False)
# plt.show()

# plt.subplot(111)
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# y1 = np.array([866, 2335, 5710, 6482, 6120, 1605, 3813, 4428, 4631])
# y2 = np.array([433, 1167, 2855, 3241, 3060, 802, 1906, 2214, 2315])

# labels = ["注册人数", "激活人数"]
# plt.stackplot(x, y1, y2, labels=labels)
# plt.title("XXX公司1-9月注册与激活人数", color='k')
# plt.xlabel('月份')
# plt.ylabel('注册与激活人数')
# plt.xticks(x, ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月"])
# plt.grid(False)
# plt.legend()
# plt.show()

# plt.subplot(111)
# x = np.array([5.5, 6.6, 8.1, 15.8, 19.5, 22.4, 28.3, 28.9])
# y = np.array([2.38, 3.85, 4.41, 5.67, 5.44, 6.03, 8.15, 6.87])
# colors = y*10  # 根据y值的大小生成不同的颜色
# area = y*100  # 根据y值的大小生成大小不同的形状

# plt.scatter(x, y, c=colors, s=area, marker='o')
# plt.title("1-8月平均气温与啤酒销量关系图", loc='center', color='k')
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='center', fontsize=11, color='white')
# plt.xlabel('平均气温')
# plt.ylabel('啤酒销量')
# plt.grid(False)
# plt.show()

# plt.subplot(111)
# x = [5.5, 6.6, 8.1, 15.8, 19.5, 22.4, 28.3, 28.9]
# y = [2.38, 3.85, 4.41, 5.67, 5.44, 6.03, 8.15, 6.87]

# plt.scatter(x, y, marker='o', s=100)
# plt.title("1-8月平均气温与啤酒销量关系图", loc='center', color='k')
# plt.xlabel('平均气温')
# plt.ylabel('啤酒销量')
# plt.grid(False)
# plt.show()

# plt.rcParams['font.family'] = "Microsoft YaHei"  # 设置支持中文字体
# plt.rcParams['axes.facecolor'] = '#E5E5E5'  # 设置背景色

# plt.subplot(111)
# x = np.array(["东区", "北区", "南区", "西区"])
# y = np.array([8566, 5335, 7310, 6482])

# plt.barh(x, height=0.5, width=y, align="center")
# plt.title("全国各分区任务量", color='k')
# for a, b in zip(x, y):
#     plt.text(b, a, b, ha='center', va='center', fontsize=11, color='k')
# plt.ylabel('区域')
# plt.xlabel('任务量')
# plt.grid(False)
# plt.show()

# x = np.array(["东区", "北区", "南区", "西区"])
# y1 = np.array([8566, 5335, 7310, 6482])
# y2 = np.array([4283, 2667, 3655, 3241])

# plt.bar(x, y1, width=0.3, label="任务量")
# plt.bar(x, y2, width=0.3, label="完成量", color='b')
# plt.title("全国各分区任务量和完成量", color='k')
# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11, color='k')
# for a, b in zip(x, y2):
#     plt.text(a, b, b, ha='center', va='top', fontsize=11, color='k')
# plt.xlabel('区域')
# plt.ylabel('任务情况')
# plt.grid(False)
# plt.legend(ncol=2, loc='upper center')
# plt.show()

# plt.subplot(111)
# x = np.array([1, 2, 3, 4])
# y1 = np.array([8566, 5335, 7310, 6482])
# y2 = np.array([4283, 2667, 3655, 3241])

# plt.bar(x, y1, width=0.3, label="任务量")
# plt.bar(x+0.3, y2, width=0.3, label="完成量", color='b')
# # x+0.3相当于把完成量的每个柱子右移0.3
# plt.title("全国各分区任务量和完成量", color='k')
# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11, color='k')
# for a, b in zip(x+0.3, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11, color='k')
# plt.xlabel('区域')
# plt.ylabel('任务情况')
# plt.xticks(x+0.15, ["东区", "南区", "西区", "北区"])
# plt.grid(False)
# plt.legend(ncol=2)
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams['font.family'] = "Microsoft YaHei"  # 设置支持中文字体
# plt.rcParams['axes.facecolor'] = '#E5E5E5'  # 设置背景色

# x = np.array(["东区", "北区", "南区", "西区"])
# y = np.array([8566, 6482, 5335, 7310])

# plt.bar(x, y, width=0.5, align='center', label="任务量", color='b')
# plt.title("全国各分区任务量", color='k')
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11, color='k')
# plt.xlabel('分区')
# plt.ylabel('任务量')
# plt.legend()

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams['font.family'] = "Microsoft YaHei"  # 设置支持中文字体
# plt.rcParams['axes.facecolor'] = '#E5E5E5'  # 设置背景色

# plt.subplot(111)
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# y = np.array([866, 2335, 5710, 6482, 6120, 1605, 3813, 4428, 4631])

# plt.plot(x, y, color='k', linestyle='-.', linewidth=1,
#          marker='o', markersize=5, label="注册用户数")
# for a, b in zip(x, y):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=11, color='k')
# plt.title("XXX公司1-9月注册用户数", color='k')
# plt.grid(True)  # 设置网格线
# plt.legend()

# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(-3, 3, 50)
# y = 2*x + 1

# plt.figure(num=1, figsize=(8, 5),)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# plt.plot(x, y,)

# x0 = 1
# y0 = 2*x0 + 1
# plt.plot([x0, x0, ], [0, y0, ], 'k--', linewidth=2.5)
# plt.scatter([x0, ], [y0, ], s=50, color='b')
# plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data',
#              xytext=(+30, -30), textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
# plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
#          fontdict={'size': 16, 'color': 'r'})
# plt.show()

# from pylab import plot, legend, gca, xticks, yticks,\
#                  title, annotate, np, scatter, show

# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# C, S = np.cos(X), np.sin(X)

# title("Matplot Demo", loc="center")
# plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine", zorder=0)
# plot(X, S, color="red",  linewidth=2.5, linestyle="-", label="sine", zorder=0)
# legend(loc='upper left')

# xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
#        [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
# yticks([-1, 0, +1], [r'$-1$', r'$0$', r'$+1$'])

# t = 2*np.pi/3
# plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--", zorder=1)
# scatter([t, ], [np.cos(t), ], 50, color='blue')

# annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$', xy=(t, np.sin(t)),
#          xycoords='data', xytext=(+10, +30), textcoords='offset points',
#          fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# plot([t, t], [0, np.sin(t)], color='red', linewidth=2.5, linestyle="--", zorder=1)
# scatter([t, ], [np.sin(t), ], 50, color='red')

# annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$', xy=(t, np.cos(t)), xycoords='data',
#          xytext=(-90, -50), textcoords='offset points', fontsize=16,
#          arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
# ax = gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))

# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)
#     label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.6, zorder=2))
# show()
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(-3, 3, 50)
# y1 = 2*x + 1
# y2 = x**2

# l1, = plt.plot(x, y1, label='linear line')
# l2, = plt.plot(x, y2, color='red', linewidth=1.0,
#                linestyle='--', label='square line')
# plt.legend(handles=[l1, l2, ], labels=['up', 'down'], loc='best')

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(-3, 3, 50)
# y1 = 2*x + 1
# y2 = x**2

# plt.figure(num=3, figsize=(8, 5))
# plt.plot(x, y2)
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xticks(np.linspace(-1, 2, 5))
# plt.yticks([-2, -1.8, -1, 1.22, 3],
#            [r'$really\ bad$', r'$bad$', r'$normal$',
#             r'$good$', r'$really\ good$'])
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# plt.show()

# from pylab import figure, subplot, plot, xlim, xticks,\
#     ylim, yticks, xlabel, ylabel, savefig, show, gca, np

# # 创建一个分辨率为 80, 长宽比为8 : 6 的图
# figure(figsize=(8, 6), dpi=80)

# # 整个figure分成1行1列，共1个子图（也可写成subplot(111)）
# subplot(1, 1, 1)

# # 定义新刻度范围以及个数：范围是(-π, π);个数是256
# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# C, S = np.cos(X), np.sin(X)

# # 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
# plot(X, C, color="blue", linewidth=1.0, linestyle="-")

# # 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
# plot(X, S, color="green", linewidth=1.0, linestyle="-")

# # 设置横轴的上下限
# xlim(-4.0, 4.0)

# # 设置纵轴的上下限
# ylim(-1.0, 1.0)

# # 设置横轴记号
# # xticks(np.linspace(-4, 4, 9, endpoint=True))

# xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
#        [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

# yticks([-1, 0, +1],
#        [r'$-1$', r'$0$', r'$+1$'])

# # 设置轴标签
# xlabel("x (-4, 4)")
# ylabel("y (-1, 1)")

# ax = gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))

# 以分辨率 72 来保存图片
# savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), 'result.png')), dpi=72)


# 在屏幕上显示
# show()

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.arange(1, 11)
# y = 2 * x + 5
# plt.rcParams['font.family'] = ['STLiti']
# plt.title("Matplot示例")
# plt.xlabel("x 轴")
# plt.ylabel("y 轴")
# plt.plot(x, y, "ob")
# plt.show()


# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np

# SourceHan = matplotlib.font_manager.FontProperties(
#     fname=os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'SourceHanSansSC-Bold.otf')))

# x = np.arange(1, 11)
# y = 2 * x + 5

# plt.subplot(2,  1,  1)
# plt.title("Matplot示例", fontproperties=SourceHan)

# plt.xlabel("x 轴", fontproperties=SourceHan)
# plt.ylabel("y 轴", fontproperties=SourceHan)
# plt.plot(x, y)

# plt.show()

# # 计算正弦和余弦曲线上的点的 x 和 y 坐标
# x = np.arange(0,  3  * np.pi,  0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)
# # 建立 subplot 网格，高为 2，宽为 1
# # 激活第一个 subplot
# plt.subplot(2,  1,  1)
# # 绘制第一个图像
# plt.plot(x, y_sin)
# plt.title('Sine')
# # 将第二个 subplot 激活，并绘制第二个图像
# plt.subplot(2,  1,  2)
# plt.plot(x, y_cos)
# plt.title('Cosine')
# # 展示图像
# plt.show()
