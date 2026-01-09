'''
Em-A model construction
'''
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import Data_Loader as data
from Metrics import MyMetrics

clf5 = RandomForestClassifier(n_estimators=15,
                              n_jobs=-1,
                              criterion='entropy',
                              random_state=666)

imp_2 = pd.read_csv('data/Em_A_lasso_result.csv')

clf5.fit(data.x_train_2[imp_2['x'].values], data.y_train_2)

predict_train = [round(i) for i in clf5.predict(data.x_train_2[imp_2['x'].values])]
predict_test = [round(i) for i in clf5.predict(data.x_test_2[imp_2['x'].values])]

sensitivity, specificity, precision, f1_score = MyMetrics(
    Predict=predict_train,
    Target=data.y_train_2,
    twoClass=True)
sensitivity_, specificity_, precision_, f1_score_ = MyMetrics(
    Predict=predict_test,
    Target=data.y_test_2,
    twoClass=True)
accuracy_train = accuracy_score(predict_train, data.y_train_2)
accuracy_test = accuracy_score(predict_test, data.y_test_2)
auc_train = roc_auc_score(predict_train, data.y_train_2)
auc_test = roc_auc_score(predict_test, data.y_test_2)

if __name__ == '__main__':
    predict_train = [round(i) for i in clf5.predict(data.x_train_2[imp_2['x'].values])]
    predict_test = [round(i) for i in clf5.predict(data.x_test_2[imp_2['x'].values])]

    sensitivity, specificity, precision, f1_score = MyMetrics(
                                                                Predict=predict_train,
                                                                Target=data.y_train_2,
                                                                twoClass=True)
    sensitivity_, specificity_, precision_, f1_score_ = MyMetrics(
                                                                Predict=predict_test,
                                                                Target=data.y_test_2,
                                                                twoClass=True)
    accuracy_train = accuracy_score(predict_train, data.y_train_2)
    accuracy_test = accuracy_score(predict_test, data.y_test_2)
    auc_train = roc_auc_score(predict_train, data.y_train_2)
    auc_test = roc_auc_score(predict_test, data.y_test_2)

    print('\n========= Em-A Metrics =========')
    print('Em-A Training   Precision   {:.3f}'.format(precision))
    print('Em-A Validation Precision   {:.3f}'.format(precision_))
    print('Em-A Training   Accuracy    {:.3f}'.format(accuracy_train))
    print('Em-A Validation Accuracy    {:.3f}'.format(accuracy_test))
    print('Em-A Training   ROC-AUC     {:.3f}'.format(auc_train))
    print('Em-A Validation ROC-AUC     {:.3f}'.format(auc_test))
    print('Em-A Training   Sensitivity {:.3f}'.format(sensitivity))
    print('Em-A Validation Sensitivity {:.3f}'.format(sensitivity_))
    print('Em-A Training   Specificity {:.3f}'.format(specificity))
    print('Em-A Validation Specificity {:.3f}'.format(specificity_))
    print('Em-A Validation F1-Score    {:.3f}'.format(f1_score_))
