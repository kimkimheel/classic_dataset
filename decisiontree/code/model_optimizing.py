from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 切割数据
features = pd.read_csv(r"C:\Users\EDZ\Desktop\classic_dataset\decisiontree\titanic\feature_df.csv")
outcomes = pd.read_csv(r"C:\Users\EDZ\Desktop\classic_dataset\decisiontree\titanic\train_label.csv")
# features = features.drop(['Sex_female', 'Sex_male'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(features,outcomes, test_size=0.2, random_state=42)

# optimize
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='gini',random_state=42)

parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10],
              'min_samples_split':[2,4,6,8,10]}

scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf,parameters,scoring=scorer)

# grid_fit = grid_obj.fit(x_train,y_train)
c, r = np.array(y_train).shape
y_train = np.array(y_train).reshape(c,)
grid_fit = grid_obj.fit(X_train,y_train)

best_fit = grid_fit.best_estimator_

best_fit.fit(X_train,y_train)

best_train_prediction = best_fit.predict(X_train)

best_test_prediction = best_fit.predict(X_test)

print('The training accuracy is', accuracy_score(best_train_prediction,y_train))

print('The test accuracy is',accuracy_score(best_test_prediction,y_test))

