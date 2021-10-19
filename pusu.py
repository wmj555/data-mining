from math import log
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import io
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB, CategoricalNB, BernoulliNB, ComplementNB


def chuli():
    data = pd.read_csv('diabetes.csv')
    X = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
    Y = data["Class"]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=1)
    return X_train,X_test,Y_train,Y_test
X_train,X_test,Y_train,Y_test = chuli()
#MultinomialNB
#特点：特征服从多项式概率分布（类似抛骰子）
#scikit-learn中mnb分类器可接受连续型特征，特征取值不能为负数。但一般输入是分类特征。
def pusu():
    mnb = MultinomialNB()
    mnb.fit(X_train,Y_train)
    y_predict = mnb.predict(X_test)
    mnb_roc_auc = mnb.score(X_test, Y_test)
    print( mnb_roc_auc)
#假定属性/特征是服从正态分布的
def GaussianNBpusu():
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    gnb_roc_auc = gnb.score(X_test, Y_test)
    print(gnb_roc_auc)
pusu()
GaussianNBpusu()
#BernoulliNB对根据多元伯努利分布分配的数据实施朴素的贝叶斯训练和分类算法；也就是说，可能有多个特征，但每个特征都假定为一个二进制值（伯努利，布尔值）变量。因此，此类要求将样本表示为二进制值特征向量。
def BernoulliNBpusu():
    bnb = BernoulliNB()
    bnb.fit(X_train, Y_train)
    bnb_roc_auc = bnb.score(X_test, Y_test)
    print(bnb_roc_auc)
BernoulliNBpusu()
#ComplementNB实现补码朴素贝叶斯（CNB）算法。CNB是标准多项式朴素贝叶斯（MNB）算法的改编，特别适合于不平衡数据集。具体来说，CNB使用来自每个类的补充的统计信息来计算模型的权重。CNB的发明人凭经验表明，CNB的参数估计比MNB的参数估计更稳定。
def ComplementNBpusu():
    tnb = ComplementNB()
    tnb.fit(X_train, Y_train)
    tnb_roc_auc = tnb.score(X_test, Y_test)
    print(tnb_roc_auc)
ComplementNBpusu()
