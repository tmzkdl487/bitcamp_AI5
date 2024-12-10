import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression # <- 분류 모델, 
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingClassifier, VotingRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=4444,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    # stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

# model = XGBRegressor()

model = VotingRegressor(
     estimators = [('XGB', xgb), ('RF', rf), ('CAT', cat)],
 )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print('최종점수 : ', results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score', r2)

# xgb
# 최종점수 :  0.3505316025455074
# r2_score 0.3505316025455074

# VotingRegressor
# 최종점수 :  0.4669082206031806
# r2_score 0.4669082206031806