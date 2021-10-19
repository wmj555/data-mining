import  pandas as  pd
import csv
import matplotlib.pyplot as plt
data = pd.read_csv('diabetes.csv')
X = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y = data["Class"]

df = data
#处理缺失值
null_all = data.isnull().sum()
print(null_all)
fig,axes = plt.subplots(1,9)

# boxes表示箱体，whisker表示触须线
# medians表示中位数，caps表示最大与最小值界限

df.plot(kind='box', ax=axes, subplots=True,
          title='Different boxplots', sym='r+')
# sym参数表示异常值标记的方式
fig.subplots_adjust(wspace=6,hspace=6)  # 调整子图之间的间距
plt.savefig('p1.png')