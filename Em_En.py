'''
New Em-En model
'''
from sklearn.metrics import accuracy_score
import Data_Loader as data
import warnings
warnings.filterwarnings('ignore')
import Em_A as SK
import Logistic_Regression as LRD
import Em_3Class as Em3
from Metrics import *

def Ensemble_Dataloder(train_data):
    step2_0_v = LRD.LR_s0.predict_proba(train_data[LRD.imp_0.columns[1:]])[:, 1]
    step2_1_v = LRD.LR_s1.predict_proba(train_data[LRD.imp_1.columns[1:]])[:, 1]
    step2_2_v = SK.clf5.predict_proba(train_data.loc[:, SK.imp_2['x'].values])[:, 1]

    x_data = train_data.loc[:, Em3.imp_SA['index'].values]
    x_data['step2_0'] = step2_0_v
    x_data['step2_1'] = step2_1_v
    x_data['step2_2'] = step2_2_v
    return x_data

def Em_en(train_data,train_target,val_data,val_target,train_mode=True):
    if train_mode:
        x_data = train_data
        y_data = train_target
        ensemble_data = Ensemble_Dataloder(x_data)
        ensemble_data = Em3.lda_3.transform(ensemble_data)
        predict_pro = []
        predict_res = []
    else:
        x_data = val_data
        y_data = val_target
        ensemble_data = Ensemble_Dataloder(x_data)
        ensemble_data = Em3.lda_3.transform(ensemble_data)
        predict_pro = []
        predict_res = []

    for i in range(len(y_data)):
        data_s1 = np.array(x_data[LRD.imp_2class.columns[1:]].iloc[i, :].values)
        step1_predict = LRD.LR_1.predict(data_s1.reshape(1, -1))[0]
        if train_mode:
            if step1_predict == 0:
                predict_res.append(step1_predict)
                step1_predict_pro = LRD.LR_1.predict_proba(data_s1.reshape(1, -1))[0]
                step1_padding = [0.0,0.0]
                step1_predict_pro_res = np.concatenate([step1_predict_pro,step1_padding],axis=0).tolist()
                predict_pro.extend(step1_predict_pro_res)
            else:
                data_s2 = ensemble_data[i].reshape(1, -1)
                step2_predict = Em3.clf5.predict(data_s2.reshape(1, -1))[0]
                step2_predict_pro = Em3.clf5.predict_proba(data_s2.reshape(1, -1))[0]
                step2_padding = [0.0]
                step2_predict_pro_res = np.concatenate([step2_padding,step2_predict_pro], axis=0).tolist()
                predict_pro.extend(step2_predict_pro_res)
                predict_res.append(step2_predict+1)
        else:
            if step1_predict == 0:
                predict_res.append(step1_predict)
                step1_predict_pro = LRD.LR_1.predict_proba(data_s1.reshape(1, -1))[0]
                step1_padding = [0.0, 0.0]
                step1_predict_pro_res = np.concatenate([step1_predict_pro, step1_padding], axis=0).tolist()
                predict_pro.extend(step1_predict_pro_res)
            else:
                data_s2 = ensemble_data[i].reshape(1, -1)
                step2_predict = Em3.clf5.predict(data_s2.reshape(1, -1))[0]
                step2_predict_pro = Em3.clf5.predict_proba(data_s2.reshape(1, -1))[0]
                step2_padding = [0.0]
                step2_predict_pro_res = np.concatenate([step2_padding, step2_predict_pro], axis=0).tolist()
                predict_pro.extend(step2_predict_pro_res)
                predict_res.append(step2_predict+1)
    return predict_res, predict_pro


if __name__ == '__main__':

    predict_train, predict_train_pro = Em_en(train_data=data.x_train_C2,
                                             train_target=data.y_train_C2,
                                             val_data=data.x_test_C2,
                                             val_target=data.y_test_C2,
                                             train_mode=True)
    predict_val, predict_val_pro = Em_en(train_data=data.x_train_C2,
                                         train_target=data.y_train_C2,
                                         val_data=data.x_test_C2,
                                         val_target=data.y_test_C2,
                                         train_mode=False)

    try:
        Fpr_EM, Tpr_EM, AUC_EM = Roc_Auc(Predict=predict_train,
                                         Target=data.y_train_ensemble)
        Fpr_, Tpr_, AUC_ = Roc_Auc(Predict=predict_val,
                                   Target=data.y_test_ensemble)
        sensitivity, specificity, precision, f1_score = MyMetrics(
                                                                Predict=predict_train,
                                                                Target=data.y_train_ensemble,
                                                                twoClass=False,
                                                                Class=4)
        sensitivity_, specificity_, precision_, f1_score_ = MyMetrics(
                                                                Predict=predict_val,
                                                                Target=data.y_test_ensemble,
                                                                twoClass=False,
                                                                Class=4)
        accuracy_train = accuracy_score(predict_train, data.y_train_ensemble)
        accuracy_test = accuracy_score(predict_val, data.y_test_ensemble)
        print('\n========= Em-En Metrics =========')
        print('Em-En Training   Precision   {:.3f}'.format(precision))
        print('Em-En Validation Precision   {:.3f}'.format(precision_))
        print('Em-En Training   Accuracy    {:.3f}'.format(accuracy_train))
        print('Em-En Validation Accuracy    {:.3f}'.format(accuracy_test))
        print('Em-En Training   ROC-AUC     {:.3f}'.format(AUC_EM))
        print('Em-En Validation ROC-AUC     {:.3f}'.format(AUC_))
        print('Em-En Training   Sensitivity {:.3f}'.format(sensitivity))
        print('Em-En Validation Sensitivity {:.3f}'.format(sensitivity_))
        print('Em-En Training   Specificity {:.3f}'.format(specificity))
        print('Em-En Validation Specificity {:.3f}'.format(specificity_))
        print('Em-En Validation F1-Score    {:.3f}'.format(f1_score_))
    except:
        print('Something Error !!')
