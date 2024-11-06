from sklearn import datasets
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load dataset
iris = datasets.load_iris()
labels = pd.DataFrame(iris.target, columns=['labels'])
data = pd.DataFrame(iris.data, columns=['Sepal length','Sepal width','Petal length','Petal width'])
data = pd.concat([data, labels], axis=1)

# Feature
feature = data[['Sepal length', 'Sepal width', 'Petal length', 'Petal width']]

# 데이터 표준화
scaler = StandardScaler()
scaled_feature = scaler.fit_transform(feature)

# DBSCAN 모델 생성 및 학습
model = DBSCAN(eps=0.5, min_samples=5)
predict = pd.DataFrame(model.fit_predict(scaled_feature), columns=['predict'])

# Noise를 'Noise'로 라벨링
r = pd.concat([pd.DataFrame(scaled_feature, columns=feature.columns), predict], axis=1)
r['predict'] = r['predict'].map(lambda x: 'Noise' if x == -1 else x)

# Seaborn 스타일 설정
sns.set(style="whitegrid")

# Pairplot with DBSCAN results
sns.pairplot(r, hue='predict', palette='Set2')
plt.show()

# Pairplot with true labels
sns.pairplot(data, hue='labels', palette='Set2')
plt.show()