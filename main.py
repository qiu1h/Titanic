import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    # 精确过滤tensorflow和tensorboard中的FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
    warnings.filterwarnings("ignore", category=FutureWarning, module="tensorboard")

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print(train.shape)
    print(test.shape)

    # 筛选存活乘客的年龄
    # survived_ages = train[train['Survived'] == 1]['Age'].dropna()
    #
    # # 绘制直方图（横轴：年龄，纵轴：存活数）
    # sns.histplot(survived_ages, bins=20, kde=False)
    # plt.xlabel('Age')
    # plt.ylabel('Number of Survivors')
    # plt.title('Distribution of Survivors by Age')
    # plt.show()

    y = train['Survived']

    train['family'] = train['SibSp'] + train['Parch'] + 1
    test['family'] = test['SibSp'] + test['Parch'] + 1

    train['SocialClass'] = train['Fare'] / train['Pclass']
    test['SocialClass'] = test['Fare'] / test['Pclass']

    train_id = train['PassengerId']
    test_id = test['PassengerId']
    train.drop(['PassengerId','Survived','Name','Ticket','Embarked','SibSp','Parch','Fare','Pclass'], axis=1, inplace=True)
    test.drop(['PassengerId','Name','Ticket','Embarked','SibSp','Parch','Fare','Pclass'], axis=1, inplace=True)

    all_data = pd.concat([train,test], axis=0)

    na_total = all_data.isnull().sum()
    na_cols = na_total[na_total > 0].sort_values(ascending=False)
    print("缺失值最多的列：\n", na_cols)

    age_mean = train['Age'].mean()
    all_data['Age'] = all_data['Age'].fillna(age_mean)
    sc_mean = train['SocialClass'].mean()
    all_data['SocialClass'] = all_data['SocialClass'].fillna(sc_mean)
    all_data['Cabin'] = all_data['Cabin'].fillna("None")

    all_data = pd.get_dummies(all_data)

    print("编码后总维度：",all_data.shape)
    print(all_data.columns.tolist())

    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(all_data)

    X = all_data_scaled[:train_id.shape[0]]
    X_test = all_data_scaled[train_id.shape[0]:]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    y = y.values
    model.fit(X, y, epochs=100, verbose=2)

    y_pred = model.predict_classes(X_test)
    print(y_pred.shape)
    y_pred_1d = y_pred.flatten()  # 转换为一维数组

    submission = pd.DataFrame({
        'Id': test_id,
        'Survived': y_pred_1d
    })
    submission.to_csv("data/submission.csv", index=False)
