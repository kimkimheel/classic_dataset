from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
sys.path.append(r'C:\Users\EDZ\Anaconda3\Graphviz\bin')

temp = os.environ['PATH']
print('Graphviz' in temp)
# 切割数据
features = pd.read_csv(r"C:\Users\EDZ\Desktop\classic_dataset\decisiontree\titanic\feature_df.csv")
outcomes = pd.read_csv(r"C:\Users\EDZ\Desktop\classic_dataset\decisiontree\titanic\train_label.csv")
x_train,x_test,y_train,y_test = train_test_split(features,outcomes, test_size=0.2, random_state=42)

# 导入分类器
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini')
model.fit(x_train, y_train)

# 测试模型
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

# 改善模型
# Training the model
model = DecisionTreeClassifier(criterion='gini',max_depth=6, min_samples_leaf=6, min_samples_split=10)
model.fit(x_train, y_train)

# Making predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Calculating accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

# 决策树可视化
import graphviz
from sklearn import tree

# 决策树可视化
dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.view()
