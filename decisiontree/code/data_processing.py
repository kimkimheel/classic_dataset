# 查看空数据情况
import pandas as pd
full_data = pd.read_csv(r"C:\Users\EDZ\Desktop\classic_dataset\decisiontree\titanic\train.csv")

print(full_data.isnull().sum())

# 拷贝一份数据出来
data_need = full_data.copy()
# ----------------------------------数据预处理 -------------------------------------------
# 年龄
data_need.Age = data_need.Age.fillna(data_need.Age.median(),axis=0)
# 去掉cabin字段
data_need.drop('Cabin', axis=1,inplace=True)
# 根据前面的描述性数据分析，embarked最多的是S，所以以S填充embarked仅有的两个空值
data_need['Embarked'].fillna('S',inplace = True)
# ----------------------------------end -------------------------------------------
# ----------------------------------特征筛选 -------------------------------------------
# 添加title字段作为一个特征
# 获取title列
data_need['title'] = data_need.Name.map(lambda x : (x.split(',')[1].split('.')[0].strip()))
# 查看处理结果
print(data_need.title.value_counts())
# 进一步分类转化
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
data_need['title'] = data_need['title'].map(title_mapDict)
# 查看处理结果
print(data_need.title.value_counts())

# 是否有同行亲属
data_need['fellow'] = data_need[['SibSp','Parch']].apply(lambda x:1 if x['SibSp'] or x['Parch'] else 0,axis=1)

# 筛选特征
features = ['Pclass','Sex', 'Age','Fare',
            'Embarked', 'title', 'fellow']
data_final = data_need[features]

# 进行独热编码处理
train_features = pd.get_dummies(data_final)

train_labels = data_need['Survived']

train_features.to_csv(r"C:\Users\EDZ\Desktop\classic_dataset\decisiontree\titanic\feature_df.csv", index=False)
train_labels.to_csv(r"C:\Users\EDZ\Desktop\classic_dataset\decisiontree\titanic\train_label.csv", index=False)