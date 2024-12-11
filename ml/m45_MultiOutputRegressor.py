import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

#1. 데이터
x, y = load_linnerud(return_X_y=True)
# print(x.shape, y.shape) # (20, 3) (20, 3)

# print(x)
# [[  5. 162.  60.] -> [[191.  36.  50.]
#  [  2. 110.  60.] 
#  [ 12. 101. 101.]
#  [ 12. 105.  37.]
#  [ 13. 155.  58.]
#  [  4. 101.  42.]
#  [  8. 101.  38.]
#  [  6. 125.  40.]
#  [ 15. 200.  40.]
#  [ 17. 251. 250.]
#  [ 17. 120.  38.]
#  [ 13. 210. 115.]
#  [ 14. 215. 105.]
#  [  1.  50.  50.]
#  [  6.  70.  31.]
#  [ 12. 210. 120.]
#  [  4.  60.  25.]
#  [ 11. 230.  80.]
#  [ 15. 225.  73.]
#  [  2. 110.  43.]]    -> [138.  33.  68.]]

# print(y)
# [[191.  36.  50.]
#  [189.  37.  52.]
#  [193.  38.  58.]
#  [162.  35.  62.]
#  [189.  35.  46.]
#  [182.  36.  56.]
#  [211.  38.  56.]
#  [167.  34.  60.]
#  [176.  31.  74.]
#  [154.  33.  56.]
#  [169.  34.  50.]
#  [166.  33.  52.]
#  [154.  34.  64.]
#  [247.  46.  50.]
#  [193.  36.  46.]
#  [202.  37.  62.]
#  [176.  37.  54.]
#  [157.  32.  52.]
#  [156.  33.  54.]
#  [138.  33.  68.]]

################  요런 데이터 얌 #################
#       x                     y
# [[  5. 162.  60.] -> [[191.  36.  50.]
# ..........................
#  [  2. 110.  43.]]-> [138.  33.  68.]]

#2. 모델
print("====================== RandomForestRegressor =========================")
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # RandomForestRegressor 스코어 :  3.5653
print(model.predict([[2, 110, 43]]))             # [[157.38  34.4   63.38]]

print("====================== LinearRegression =========================")

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # LinearRegression 스코어 :  7.4567
print(model.predict([[2, 110, 43]]))             # [[187.33745435  37.08997099  55.40216714]]

print("=====================  Ridge ==========================")

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # Ridge 스코어 :  7.4569
print(model.predict([[2, 110, 43]]))             # [[187.32842123  37.0873515   55.40215097]]

print("=====================  XGBRegressor ==========================")

model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # XGBRegressor 스코어 :  0.0008
print(model.predict([[2, 110, 43]]))             # [[138.0005    33.002136  67.99897 ]]

print("=====================  CatBoostRegressor 에러남.  ==========================")

# model = CatBoostRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 : ',
#       round(mean_absolute_error(y, y_pred), 4))  
# print(model.predict([[2, 110, 43]]))             
# Traceback (most recent call last):
#   File "c:\ai5\study\ml\m45_MultiOutputRegressor.py", line 106, in <module>
#     model.fit(x, y)
#   File "C:\anaconda3\envs\tf274gpu\lib\site-packages\catboost\core.py", line 5873, in fit
#     return self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline,
#   File "C:\anaconda3\envs\tf274gpu\lib\site-packages\catboost\core.py", line 2410, in _fit
#     self._train(
#   File "C:\anaconda3\envs\tf274gpu\lib\site-packages\catboost\core.py", line 1790, in _train
#     self._object._train(train_pool, test_pool, params, allow_clear_pool, init_model._object if init_model else None)
#   File "_catboost.pyx", line 5017, in _catboost._CatBoost._train
#   File "_catboost.pyx", line 5066, in _catboost._CatBoost._train
# _catboost.CatBoostError: catboost/private/libs/target/data_providers.cpp:639: Currently only multi-regression, multilabel and survival objectives work with multidimensional target

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
# import warnings
# warnings.filterwarnings('ignore') # <- 안먹혀서 주석처리함.

model =MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # MultiOutputRegressor 스코어 :  8.91
print(model.predict([[2, 110, 43]]))             # [[178.6  35.4  56.1]]
# MultiOutputRegressor안감싸면 에러남. ValueError: y should be a 1d array, got an array of shape (20, 3) instead.

model =MultiOutputRegressor(CatBoostRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # MultiOutputRegressor 스코어 :  0.2154
print(model.predict([[2, 110, 43]]))             # [[138.97756017  33.09066774  67.61547996]]

model =CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))  # MultiOutputRegressor 스코어 :  0.2154
print(model.predict([[2, 110, 43]]))             # [[138.97756017  33.09066774  67.61547996]]
