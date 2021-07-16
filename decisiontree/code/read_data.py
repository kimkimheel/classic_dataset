import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from pylab import mpl

# mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["font.sans-serif"]="SimHei"     #解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False


full_data = pd.read_csv(r"C:\Users\EDZ\Desktop\decisiontree\titanic\train.csv")

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis = 1)

# # 查看空值
print(features_raw.isnull().sum())
#
# # 查看存活人数
survived_count = outcomes.value_counts()  # !!! important


fig1 = plt.figure(figsize=(6,6))
plt.pie(list(survived_count),labels=['未存活','存活'],autopct='%1.1f%%',startangle=100)
fig1.show()

# # 每一个特征对存活的影响
# pclass

fig2 = plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.countplot(x=full_data.Pclass, hue=full_data.Survived, palette='Blues')
fig2.show()

# sex
fig3 = plt.figure(figsize=(6,6))
sns.set_theme(style='whitegrid')
sns.countplot(x=full_data.Sex, hue=full_data.Survived, palette='Blues')
fig3.show()

# Embarked
fig4 = plt.figure(figsize=(6,6))
sns.set_theme(style='whitegrid')
sns.countplot(x=full_data.Embarked, hue=full_data.Survived, palette='Blues')
fig4.show()

# age
fig5 = plt.figure(figsize=(6,6))
sns.set_theme(style='whitegrid')
sns.boxplot(x=full_data.Survived, y = full_data.Age,palette='Blues')
fig5.show()

# fare
fig6 = plt.figure(figsize=(6,6))
sns.set_theme(style='whitegrid')
sns.violinplot(x=full_data.Survived, y = full_data.Fare,palette='Blues')
fig6.show()

# sibsp
full_data['sibsp'] = full_data['SibSp'].map(lambda x:1 if x else 0)
fig7 = plt.figure(figsize=(6,6))
sns.set_theme(style='whitegrid')
sns.countplot(x=full_data.sibsp, hue=full_data.Survived, palette='Blues')
fig7.show()

# parch
full_data['parch'] = full_data['Parch'].map(lambda x:1 if x else 0)
fig8 = plt.figure(figsize=(6,6))
sns.set_theme(style='whitegrid')
sns.countplot(x=full_data.parch, hue=full_data.Survived, palette='Blues')
fig8.show()

# fellow
full_data['fellow'] = full_data[['SibSp','Parch']].apply(lambda x:1 if x['SibSp'] or x['Parch'] else 0,axis=1)
fig9 = plt.figure(figsize=(6,6))
sns.set_theme(style='whitegrid')
sns.countplot(x=full_data.fellow, hue=full_data.Survived, palette='Blues')
fig9.show()