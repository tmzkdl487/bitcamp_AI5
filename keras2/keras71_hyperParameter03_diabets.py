import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') 

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=336,
                                                    train_size=0.8,
                                                    # stratify=y,
                                                    )

# print(x_train.shape, y_train.shape) # (353, 10) (353,)

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001,):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name= 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name= 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name= 'hidden4')(x)
    x = Dense(node5, activation=activation, name= 'hidden5')(x)
    outputs = Dense(1, activation='linear', name= 'outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mae'],
                  loss='mse')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16] 
    node4 = [128, 64, 32, 16] 
    node5 = [128, 64, 32, 16, 8]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'node4' : node4,
            'node5' : node5,            
            } 
    
hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1,)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=5,
                           n_iter=10, 
                        #    n_jobs=-1,
                           verbose=1,
                           )

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=100)
end_time = time.time()

print("걸린시간 : ", round(end_time - start_time, 2))
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score_', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

# 걸린시간 :  72.21
# model.best_params_ {'optimizer': 'rmsprop', 
# 'node5': 32, 'node4': 64, 'node3': 32, 'node2': 64, 'node1': 16, 
# 'drop': 0.3, 'batch_size': 200, 'activation': 'linear'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasRegressor object at 0x0000022754D63BB0>
# model.best_score_ -3032.7751953125
# model.score :  -2995.77099609375