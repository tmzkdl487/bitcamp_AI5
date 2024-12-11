import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

np.random.seed(42) 

# 함수 호출 및 데이터 출력

def create_multiclass_data_with_labels():
    x = np.random.rand(20, 3)
    y = np.random.randint(0, 5, size=(20, 3))
    
    X_df = pd.DataFrame(x, columns=['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Lebel2', 'Label3'])
    
    return X_df, y_df

x, y = create_multiclass_data_with_labels()
print("X 데이터:")
print(x)
# X 데이터:
#     Feature1  Feature2  Feature3
# 0   0.374540  0.950714  0.731994
# 1   0.598658  0.156019  0.155995
# 2   0.058084  0.866176  0.601115
# 3   0.708073  0.020584  0.969910
# 4   0.832443  0.212339  0.181825
# 5   0.183405  0.304242  0.524756
# 6   0.431945  0.291229  0.611853
# 7   0.139494  0.292145  0.366362
# 8   0.456070  0.785176  0.199674
# 9   0.514234  0.592415  0.046450
# 10  0.607545  0.170524  0.065052
# 11  0.948886  0.965632  0.808397
# 12  0.304614  0.097672  0.684233
# 13  0.440152  0.122038  0.495177
# 14  0.034389  0.909320  0.258780
# 15  0.662522  0.311711  0.520068
# 16  0.546710  0.184854  0.969585
# 17  0.775133  0.939499  0.894827
# 18  0.597900  0.921874  0.088493
# 19  0.195983  0.045227  0.325330

print("\nY데이터:")
print(y)
# Y데이터:
#     Label1  Lebel2  Label3
# 0        4       1       4
# 1        1       0       3
# 2        3       3       4
# 3        0       4       4
# 4        0       0       0
# 5        0       3       2
# 6        2       0       2
# 7        2       0       2
# 8        4       1       1
# 9        0       3       0
# 10       3       1       0
# 11       4       2       3
# 12       2       2       0
# 13       2       4       2
# 14       0       4       1
# 15       2       0       1
# 16       1       3       4
# 17       2       0       3
# 18       4       3       4
# 19       4       2       4

#2. 모델
print("====================== RandomForestRegressor =========================")
model = RandomForestClassifier()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # RandomForestClassifier 스코어 :  0.0
print(model.predict([[0.195983, 0.045227, 0.325330]]))             # [[4 2 4]]

print("====================== LinearRegression =========================")

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # LinearRegression 스코어 :  1.1171
print(model.predict([[0.195983, 0.045227, 0.325330]]))             # [[1.32244673 2.10017712 1.66236438]]

print("=====================  Ridge ==========================")

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # Ridge 스코어 :  1.1592
print(model.predict([[0.195983, 0.045227, 0.325330]]))             # [[1.52153908 1.94525043 1.79423897]]

print("=====================  XGBRegressor ==========================")

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # XGBRegressor 스코어 :  0.0007
print(model.predict([[0.195983, 0.045227, 0.325330]]))             # [[3.9804282 2.0002153 3.4952357]]

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
# import warnings
# warnings.filterwarnings('ignore') # <- 안먹혀서 주석처리함.

# model = MultiOutputClassifier(LGBMClassifier())
# model.fit(x, y)
# y_pred = model.predict(x)
# print("Classification Accuracy:")
# print(accuracy_score(y, y_pred))
# print(model.predict([[0.195983, 0.045227, 0.325330]]))             # 

# model =MultiOutputClassifier(CatBoostClassifier())
# model.fit(x, y)
# y_pred = model.predict(x)
# print("Classification Accuracy:")
# print(accuracy_score(y, y_pred))
# print(model.predict([[0.195983, 0.045227, 0.325330]]))             # 

# model =CatBoostClassifier(loss_function='MultiRMSE')
# model.fit(x, y)
# y_pred = model.predict(x)
# print("Classification Accuracy:")
# print(accuracy_score(y, y_pred))
# print(model.predict([[0.195983, 0.045227, 0.325330]]))             # 

model = MultiOutputClassifier(XGBClassifier())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # MultiOutputClassifier 스코어 :  0.0333
print(model.predict([[0.195983, 0.045227, 0.325330]])) # [[4 2 4]]

model =MultiOutputClassifier(CatBoostClassifier())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred.reshape(20, 3)), 4))  # MultiOutputClassifier 스코어 :  0.0
print(model.predict([[0.195983, 0.045227, 0.325330]]))  # [[[4 2 4]]]