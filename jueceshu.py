from math import log
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import io
import matplotlib.pyplot as plt
def chuli():
    data = pd.read_csv('diabetes.csv')
    X = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
    Y = data["Class"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=1)
    return X_train,X_test,Y_train,Y_test
X_train,X_test,Y_train,Y_test = chuli()
def juece(k):
    # 决策树
    # max_depth定义树的深度, 可以用来防止过拟合
    # min_weight_fraction_leaf 定义叶子节点最少要包含多少个样本(百分比表达), 防止过拟合
    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=k)
    dtree = dtree.fit(X_train, Y_train)
    print('\n\n---决策树---')
    dt_roc_auc = roc_auc_score(Y_test, dtree.predict(X_test))
    print('决策树 AUC = %2.2f' % dt_roc_auc)
    dot_data = io.StringIO()
    tree.export_graphviz(dtree, out_file=dot_data,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")
    return dt_roc_auc
juece(3)
def juecepit():
    ks = [2,3,4,5,6,7,8,9,10,11,12]
    scores = []
    for k in ks:
        scores.append(juece(k))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, scores)
    ax.set_xlabel(r"k")
    ax.set_ylabel(r"score")
    ax.set_title("k的取值")
    plt.savefig("p2juece_k.png")
    plt.close()
#juecepit()
# dot_data = io.StringIO()
# tree.export_graphviz(dtree, out_file=dot_data,  # 绘制决策树
#
#                 filled=True, rounded=True,
#                          special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_pdf("tree.pdf")
#自己编
#决策树
def calcShannonEnt(X,Y):
    dlen = len(X)   #数据行数
    labelCounts = {}  # 创建保存每个标签出现次数的字典
    for i in range (0,dlen):  # 对每组特征向量进行统计
        currentLabel = Y.iloc[i]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / dlen  # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt
#print(calcShannonEnt(X_train,Y_train))

#按特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
            print(reducedFeatVec)
    return retDataSet
#X_train_data,X_test_data,Y_train_data,Y_test_data = np.array(X_train,X_test,Y_train,Y_test) #先将数据框转换为数组
X_train_data = np.array(X_train) #先将数据框转换为数组
train_data_list = X_train_data.tolist()  #其次转换为列表
