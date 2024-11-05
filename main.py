from sklearn import datasets
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot  as plt
import seaborn as sns

iris = datasets.load_iris()
labels = pd.DataFrame(iris.target)
labels.columns=['labels']
data = pd.DataFrame(iris.data)
data.columns=['Sepal length','Sepal width','Petal length','Petal width']
data = pd.concat([data,labels],axis=1)

feature = data[ ['Sepal length','Sepal width','Petal length','Petal width']]

# create model and prediction
model = DBSCAN(eps=0.5,min_samples=5)
predict = pd.DataFrame(model.fit_predict(feature))
predict.columns=['predict']

# concatenate labels to df as a new column
r = pd.concat([feature,predict],axis=1)

print(r)

#pairplot with Seaborn
sns.pairplot(r,hue='predict')
plt.show()

#pairplot with Seaborn
sns.pairplot(data,hue='labels')
plt.show()