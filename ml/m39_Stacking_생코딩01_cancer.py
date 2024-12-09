# Pseudo Labelling 기법 : 모델을 돌려서 나온 결과로 결측치를 찾아
# 스태킹: 모델을 돌려 나온거로 컬럼을 구성해서 새로운 데이터셋을 만듬.
#         한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼 맹그러

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # <- 분류 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=1186,  # 1199도 스태킹 0.99나옴.
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
print(y_train.shape, y_test.shape)  # (455,) (114,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

train_list = []
test_list = []

models = [xgb, rf, cat]

for model in models : 
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    y_test_predict = model. predict(x_test)
    
    train_list.append(y_predict)
    test_list.append(y_test_predict)
    
    score = accuracy_score(y_test, y_test_predict)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, score))
    
# XGBClassifier ACC : 0.9912
# RandomForestClassifier ACC : 0.9912
# CatBoostClassifier ACC : 0.9912

# print('넘파이 : ', np.__version__)  # 넘파이 :  1.20.3나옴. 선생님 넘파이 1.19.5
# 넘파이 버전 오류나면 알아서 고쳐.

x_train_new = np.array(train_list).T
print(x_train_new.shape)    # (3, 455) -> .T하면 -> (455, 3) 이렇게 바뀜.

x_test_new = np.array(test_list).T
print(x_test_new.shape) # (114, 3)

#2. 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)

y_pred = model2.predict(x_test_new)

score2 = accuracy_score(y_test, y_pred)

print("스태킹결과 : ", score2)

# 스태킹결과 :  0.9912280701754386