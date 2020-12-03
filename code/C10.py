# 乳腺癌诊断分类

# 1. 引入项目要用到的程序类库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 2. 加载数据集
data = pd.read_csv("code/data.csv")

# 3. 探索性数据分析
pd.set_option('display.max_columns', None)
print(data.columns.values)
print(data.head(4))
print(data.describe())

# 4. 将特征字段分成3组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

# 5. 数据清洗
# 数据清洗—删除ID列
data.drop("id", axis=1, inplace=True)
# B良性替换为0，M恶性替换为1
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# 6. 将肿瘤诊断结果可视化
sns.countplot(x='diagnosis', data=data, label="Count")
# plt.show()

plt.savefig("diagnosis-countplot.jpg")

# 7. 用热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14, 14))
sns.set(font_scale=0.8)
# annot=True显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()
# plt.savefig("diagnosis-heatmap.jpg")


# 8. 特征选择
features_remain = ['radius_mean', 'texture_mean', 'smoothness_mean',
                   'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']

# 9. 准备训练数据：抽取30%的数据作为测试集，其余作为训练集
train, test = train_test_split(data, test_size=0.3)
train_x = train[features_remain]
train_y = train['diagnosis']
test_x = test[features_remain]
test_y = test['diagnosis']

# 10. 采用Z-Score规范化数据：保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

# 11. 创建SVM分类器并进行训练和预测
model = svm.SVC()
# 用训练集做训练
model.fit(train_x, train_y)
# 用测试集做预测
prediction = model.predict(test_x)
print('准确率: ', metrics.accuracy_score(prediction, test_y))


# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn import datasets
# import numpy as np
# from pprint import PrettyPrinter


# def classify(title, x, y):
#     # 进行网格搜索
#     clf = GridSearchCV(SVC(random_state=42, max_iter=100), {
#                        'kernel': ['linear', 'poly', 'rbf'], 'C': [1, 10]})
#     clf.fit(x, y)
#     print("{}支持向量机准确率: {}".format(title, clf.score(x, y)))
#     PrettyPrinter().pprint(clf.cv_results_)


# rain = np.load('code/rain.npy')
# dates = np.load('code/doy.npy')

# x = np.vstack((dates[:-1], rain[:-1]))
# y = np.sign(rain[1:])

# classify('气象数据', x.T, y)

# # iris example
# iris = datasets.load_iris()
# x = iris.data[:, :2]
# y = iris.target
# classify('鸢尾花数据', x, y)


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold
# from sklearn import datasets
# import numpy as np


# def classify(title, x, y):
#     clf = LogisticRegression(random_state=12)
#     scores = []
#     # 将数据集随机拆分10份
#     kf = KFold(n_splits=10)
#     for train, test in kf.split(x):
#         clf.fit(x[train], y[train])
#         scores.append(clf.score(x[test], y[test]))

#     print("{}逻辑回归平均准确率: {}".format(title, np.mean(scores)))


# rain = np.load('code/rain.npy')
# dates = np.load('code/doy.npy')

# x = np.vstack((dates[:-1], rain[:-1]))
# y = np.sign(rain[1:])
# classify('气象数据', x.T, y)

# iris = datasets.load_iris()
# x = iris.data[:, :2]
# y = iris.target
# classify('鸢尾花数据', x, y)

# import numpy as np
# from sklearn import preprocessing
# from scipy.stats import anderson

# rain = np.load('code/rain.npy')
# rain = .1 * rain
# rain[rain < 0] = .05/2
# print("Rain 期望值: ", rain.mean())
# print("Rain 标准差: ", rain.var())
# print("Anderson-Darling检验结果: ", anderson(rain))

# scaled = preprocessing.scale(rain)
# print("缩放后期望值: ", scaled.mean())
# print("缩放后标准差: ", scaled.var())
# print("缩放后Anderson-Darling检验结果: ", anderson(scaled))

# # 特将征值由数值型转换为布尔型
# binarized = preprocessing.binarize(rain.reshape(-1, 1))
# print(np.unique(binarized), binarized.sum())
# # 用LabelBinarizer类来标注类别
# lb = preprocessing.LabelBinarizer()
# lb.fit(rain.astype(int))
# print(lb.classes_)
