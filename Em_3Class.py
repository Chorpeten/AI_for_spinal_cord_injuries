from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import Data_Loader as data
from Metrics import MyMetrics
import Logistic_Regression as LRD
import Em_A as SK

imp_SA = pd.read_csv('result/SA_bool.csv')
print(pd.Series(imp_SA['index'].values).value_counts())
def Ensemble_Dataloder(train_data):
    step2_0_v = LRD.LR_s0.predict_proba(train_data[LRD.imp_0.columns[1:]])[:, 1]
    step2_1_v = LRD.LR_s1.predict_proba(train_data[LRD.imp_1.columns[1:]])[:, 1]
    step2_2_v = SK.clf5.predict_proba(train_data.loc[:, SK.imp_2['x'].values])[:, 1]

    x_data = train_data.loc[:, imp_SA['index'].values]
    x_data['step2_0'] = step2_0_v
    x_data['step2_1'] = step2_1_v
    x_data['step2_2'] = step2_2_v
    return x_data

lda_3 = LDA(n_components=2)
data_pre = data.step2_total_data
data_pre = data_pre.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)
target_3class = data.target_s2t

data_3class = Ensemble_Dataloder(data_pre)
lda_3.fit(data_3class,target_3class)
data_3class = lda_3.transform(data_3class)

x_train, x_test, y_train, y_test = train_test_split(data_3class,
                                                    target_3class,
                                                    test_size=0.2,
                                                    random_state=29)

clf5 = RandomForestClassifier(n_estimators=1,
                              n_jobs=-1,
                              criterion='entropy',
                              class_weight={0:1.1, 1:1.3, 2:1},
                              random_state=6
                              )

clf5.fit(x_train, y_train)

if __name__ == '__main__':
    predict_train = [round(i) for i in clf5.predict(x_train)]
    predict_test = [round(i) for i in clf5.predict(x_test)]

    lb = preprocessing.LabelBinarizer()
    p_onehot_train = lb.fit_transform(predict_train).ravel()
    p_onehot_test = lb.fit_transform(predict_test).ravel()
    t_onehot_train = lb.fit_transform(y_train).ravel()
    t_onehot_test = lb.fit_transform(y_test).ravel()

    precision_train = precision_score(predict_train, y_train, average='weighted')
    precision_test = precision_score(predict_test, y_test, average='weighted')
    accuracu_train = accuracy_score(predict_train, y_train)
    accuracu_test = accuracy_score(predict_test, y_test)
    auc_train = roc_auc_score(p_onehot_train, t_onehot_train)
    auc_test = roc_auc_score(p_onehot_test, t_onehot_test)

    sensitivity, specificity, precision, f1_score = MyMetrics(
        Predict=p_onehot_test,
        Target=t_onehot_test,
        twoClass=False,
        Class=3)
    print('\n========= Em-3Class Metrics =========')
    print('Em-3Class Validation Sensitivity {:.3f}'.format(sensitivity))
    print('Em-3Class Validation Specificity {:.3f}'.format(specificity))
    print("Em-3Class Training   Precision : %0.2f " % (precision_train))
    print("Em-3Class Validation Precision : {:.3f} ".format(precision_test))
    print("Em-3Class Training   Accuracy  : %0.2f " % (accuracu_train))
    print("Em-3Class Validation Accuracy  : {:.3f}".format(accuracu_test))
    print("Em-3Class Training   ROC-Auc   : %0.2f " % (auc_train))
    print("Em-3Class Validation ROC-AUC   : {:.3f} ".format(auc_test))
    print("Em-3Class Validation F1-Score  : {:.3f} ".format(f1_score))
