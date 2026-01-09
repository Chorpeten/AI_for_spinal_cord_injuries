'''
Logistic Regression models construction
'''
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score,roc_auc_score
import pandas as pd
import Data_Loader as data
from Metrics import MyMetrics

####################################### LR step1 (Em-2Class) #######################################
imp_2class = pd.read_csv('result/ASIA_2Classes_feature_new.csv')

LR_1 = LR(
    solver='liblinear',
    C=3.5,
    random_state=888,
    max_iter=1000,
    class_weight={0: 1.5, 1: 1}
)

LR_1.fit(data.x_train_C2[imp_2class.columns[1:]], data.y_train_C2)

if __name__ == '__main__':
    predict_train = LR_1.predict(data.x_train_C2[imp_2class.columns[1:]])
    predict_test = LR_1.predict(data.x_test_C2[imp_2class.columns[1:]])
    sensitivity, specificity, precision, f1_score = MyMetrics(
        Predict=predict_train,
        Target=data.y_train_C2,
        twoClass=True)
    sensitivity_, specificity_, precision_, f1_score_ = MyMetrics(
        Predict=predict_test,
        Target=data.y_test_C2,
        twoClass=True)
    accuracy_train = accuracy_score(predict_train, data.y_train_C2)
    accuracy_test = accuracy_score(predict_test, data.y_test_C2)
    auc_train = roc_auc_score(predict_train, data.y_train_C2)
    auc_test = roc_auc_score(predict_test, data.y_test_C2)

    print('\n========= Em-2Class Metrics =========')
    print('Em-2Class Validation Precision   {:.3f}'.format(precision_))
    print('Em-2Class Validation Accuracy    {:.3f}'.format(accuracy_test))
    print('Em-2Class Validation ROC-AUC     {:.3f}'.format(auc_test))
    print('Em-2Class Validation Sensitivity {:.3f}'.format(sensitivity_))
    print('Em-2Class Validation Specificity {:.3f}'.format(specificity_))
    print('Em-2Class Validation F1-Score    {:.3f}'.format(f1_score_))

####################################### LR step2 #######################################
#################### Em-C ####################
imp_0 = pd.read_csv('result/ASIA_C_feature.csv')
LR_s0 = LR(
    solver='liblinear',
    C=0.01,
    random_state=420,
    max_iter=1000,
    class_weight={0: 1, 1: 2.3}
)

LR_s0.fit(data.x_train_0[imp_0.columns[1:]], data.y_train_0)

if __name__ == '__main__':
    predict_train = LR_s0.predict(data.x_train_0[imp_0.columns[1:]])
    predict_test = LR_s0.predict(data.x_test_0[imp_0.columns[1:]])
    sensitivity, specificity, precision, f1_score = MyMetrics(
        Predict=predict_train,
        Target=data.y_train_0,
        twoClass=True)
    sensitivity_, specificity_, precision_, f1_score_ = MyMetrics(
        Predict=predict_test,
        Target=data.y_test_0,
        twoClass=True)
    accuracy_train = accuracy_score(predict_train, data.y_train_0)
    accuracy_test = accuracy_score(predict_test, data.y_test_0)
    auc_train = roc_auc_score(predict_train, data.y_train_0)
    auc_test = roc_auc_score(predict_test, data.y_test_0)

    print('\n========= Em-C Metrics =========')
    print('Em-C Validation Precision   {:.3f}'.format(precision_))
    print('Em-C Validation Accuracy    {:.3f}'.format(accuracy_test))
    print('Em-C Validation ROC-AUC     {:.3f}'.format(auc_test))
    print('Em-C Validation Sensitivity {:.3f}'.format(sensitivity_))
    print('Em-C Validation Specificity {:.3f}'.format(specificity_))
    print('Em-C Validation F1-Score    {:.3f}'.format(f1_score_))

#################### Em-B ####################
imp_1 = pd.read_csv('result/ASIA_B_feature.csv')
LR_s1 = LR(
    solver='liblinear',
    C=0.1,
    random_state=420,
    max_iter=1000,
    class_weight={0: 1, 1: 1}
)

LR_s1.fit(data.x_train_1[imp_1.columns[1:]], data.y_train_1)

if __name__ == '__main__':
    predict_train = LR_s1.predict(data.x_train_1[imp_1.columns[1:]])
    predict_test = LR_s1.predict(data.x_test_1[imp_1.columns[1:]])
    sensitivity, specificity, precision, f1_score = MyMetrics(
        Predict=predict_train,
        Target=data.y_train_1,
        twoClass=True)
    sensitivity_, specificity_, precision_, f1_score_ = MyMetrics(
        Predict=predict_test,
        Target=data.y_test_1,
        twoClass=True)
    accuracy_train = accuracy_score(predict_train, data.y_train_1)
    accuracy_test = accuracy_score(predict_test, data.y_test_1)
    auc_train = roc_auc_score(predict_train, data.y_train_1)
    auc_test = roc_auc_score(predict_test, data.y_test_1)

    print('\n========= Em-B Metrics =========')
    print('Em-B Validation Precision   {:.3f}'.format(precision_))
    print('Em-B Validation Accuracy    {:.3f}'.format(accuracy_test))
    print('Em-B Validation ROC-AUC     {:.3f}'.format(auc_test))
    print('Em-B Validation Sensitivity {:.3f}'.format(sensitivity_))
    print('Em-B Validation Specificity {:.3f}'.format(specificity_))
    print('Em-B Validation F1-Score    {:.3f}'.format(f1_score_))
