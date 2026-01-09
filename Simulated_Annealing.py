'''
Simulated Annealing features selection
'''
import sys
import pandas as pd
sys.path.append("..")
import Data_Loader as data
import numpy as np
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from SA import Simulated_Annealing

x_train,y_train,x_test,y_test = data.x_train_3, data.y_train_3,data.x_test_3,data.y_test_3

np.random.seed(1234)
idx = np.random.permutation(len(x_train))
idx_v = np.random.permutation(len(x_test))
x_train,y_train = np.array(x_train)[idx],y_train[idx]
x_test,y_test = np.array(x_test)[idx_v],y_test[idx_v]

random.seed()
np.random.seed()

clf = ExtraTreesClassifier(n_estimators=25)
selector = Simulated_Annealing(loss_func = log_loss, estimator = clf,
                               init_temp = 0.2, min_temp = 0.005, iteration = 10,
                               alpha = 0.9, predict_type = 'predict_proba')

selector.fit(X_train = x_train, y_train = y_train, X_val = x_test,
             y_val = y_test, stop_point = 15, cv=5)

transformed_train = selector.transform(x_train)
transformed_test = selector.transform(x_test)

index = pd.DataFrame({
    'index':selector.best_sol
})
index.to_csv('result/SA_bool.csv')
