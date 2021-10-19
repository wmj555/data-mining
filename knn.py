import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('diabetes.csv')
X = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y = data["Class"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=1)


#手动编程
#求点到点的距离函数
def distance(d1,d2):
    res = 0
    for key in ("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"):
        res += (d1[key]-d2[key])**2
    return res**0.5

#编写knn
def knn(data,k):
    #1.距离
    res = [
        {"result":Y_train.iloc[i],
         "distance":distance(data,X_train.iloc[i])}
        for i in range(0,len(Y_train)-1)
    ]
    #print(res)

    #2.排序
    res = sorted(res,key=lambda item:item['distance'])
    #print(res)

    #取前k个
    res1 = res[0:k]
    #print(res1)

    #加权平均
    result = {1:0,0:0}
    #总距离
    sum = 0
    for i in res1:
        sum+=i["distance"]
    for i in res1:
        result[i['result']] +=1 - i['distance']/sum
    #print(result)

    #判定
    if result[0]>result[1]:
        return 0
    else:
        return 1

def score(k):
    #测试
    corr = 0
    for i in range(0,len(Y_test)):
        if knn(X_test.iloc[i],k)==Y_test.iloc[i]:
            corr +=1
    score = corr/len(Y_test)
    return score
    #print("正确率是：{:.2f}%".format(score))
print(score(7))
#可视化寻找最佳k值

def test_k(*data):
    ks = [2,3,4,5,6,7,8,9,10]
    scores = []
    for k in ks:
        scores.append(score(k))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, scores)
    ax.set_xlabel(r"k")
    ax.set_ylabel(r"score")
    ax.set_title("k的取值")
    plt.savefig("p0knn_k.png")
    plt.close()
#test_k()

#调用库函数，进行knn分类
def knn_ku(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    test_score = knn.score(X_test, Y_test)
    print(" test score: {}".format(test_score))
    return test_score


def test_k1(*data):
    ks = [1,2,3,4,5,6,7,8,9,10,11,12]
    scores = []
    for k in ks:
        scores.append(knn_ku(k))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, scores)
    ax.set_xlabel(r"k")
    ax.set_ylabel(r"score")
    ax.set_title("k的取值")
    plt.savefig("p1knn_k.png")
    plt.close()
#test_k1()
print(knn_ku(7))