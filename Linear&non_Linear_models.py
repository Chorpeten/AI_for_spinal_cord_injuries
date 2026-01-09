from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
import matplotlib.pyplot as plt
import Data_Loader as data
import xgboost as xgb
from Metrics import *
import Params as params
import Test as EM

####### load validation datasets
lasso = pd.read_csv('data/lasso_result.csv')
x_train = data.x_train_C2[lasso['x'].values]
y_train = data.y_train_ensemble
x_test = data.x_test_C2[lasso['x'].values]
y_test = data.y_test_ensemble

###################################### Random Forest ######################################
RF = RandomForestClassifier(n_estimators=20, random_state=2020).fit(x_train, y_train)
predict = RF.predict(x_test)
predict_pro = RF.predict_proba(x_test)

sensitivity,specificity,precision,f1_score = MyMetrics(Predict=predict,
                                                       Target=y_test,
                                                       twoClass=False,
                                                       Class=4)
Fpr_RF,Tpr_RF,AUC_RF = Roc_Auc(Predict=predict,
                            Target=y_test)

thresh_group = np.arange(0,1,0.01)
net_benefit_RF = calculate_net_benefit_model(thresh_group, predict_pro, y_test)
print('RandomForest Validation set ROC-AUC:     %.3f'% AUC_RF)

###################################### GBDT ######################################
GBDT = GradientBoostingClassifier(n_estimators=50, random_state=6).fit(x_train, y_train)
predict = GBDT.predict(x_test)
predict_pro = GBDT.predict_proba(x_test)
sensitivity,specificity,precision,f1_score = MyMetrics(Predict=predict,
                                                       Target=y_test,
                                                       twoClass=False,
                                                       Class=4)
Fpr_GBDT,Tpr_GBDT,AUC_GBDT = Roc_Auc(Predict=predict,
                                     Target=y_test)
net_benefit_GBDT = calculate_net_benefit_model(thresh_group, predict_pro, y_test)
print('GBDT Validation set ROC-AUC:     %.3f'% AUC_GBDT)

###################################### SVM ######################################
SVM = svm.SVC(decision_function_shape='ovo',
              random_state=6,
              kernel='poly',
              C=2).fit(x_train, y_train)
predict = SVM.predict(x_test)
SVM = svm.SVC(decision_function_shape='ovo',
              random_state=6,
              kernel='poly',
              C=2,
              probability=True).fit(x_train, y_train)
predict_pro = SVM.predict_proba(x_test)
sensitivity,specificity,precision,f1_score = MyMetrics(Predict=predict,
                                                       Target=y_test,
                                                       twoClass=False,
                                                       Class=4)
Fpr_SVM,Tpr_SVM,AUC_SVM = Roc_Auc(Predict=predict,
                                Target=y_test)
net_benefit_SVM = calculate_net_benefit_model(thresh_group, predict_pro, y_test)
print('SVM Validation set ROC-AUC:     %.3f'% AUC_SVM)

###################################### XGBoost ######################################
xgtrain = xgb.DMatrix(x_train, y_train)
xgtest = xgb.DMatrix(x_test, y_test)
XGBoost = xgb.train(params.params_xgboost,
                    xgtrain,
                    num_boost_round = 100)
predict = XGBoost.predict(xgtest)
XGBoost = xgb.train(params.params_prob,
                    xgtrain,
                    num_boost_round = 100)
predict_pro = XGBoost.predict(xgtest)

sensitivity,specificity,precision,f1_score = MyMetrics(Predict=predict,
                                                       Target=y_test,
                                                       twoClass=False,
                                                       Class=4)
Fpr_XGBoost,Tpr_XGBoost,AUC_XGBoost = Roc_Auc(Predict=predict,
                                              Target=y_test)
net_benefit_XGBoost = calculate_net_benefit_model(thresh_group, predict_pro, y_test)
print('XGBoost Validation set ROC-AUC:     %.3f'% AUC_XGBoost)

###################################### MLR ######################################
MLR = LR(
    solver='saga',
    C=2,
    random_state=20,
    multi_class='multinomial',
    max_iter=10000
).fit(x_train ,y_train)

predict = MLR.predict(x_test)
predict_pro = MLR.predict_proba(x_test)
sensitivity,specificity,precision,f1_score = MyMetrics(Predict=predict,
                                                       Target=y_test,
                                                       twoClass=False,
                                                       Class=4)
Fpr_MLR,Tpr_MLR,AUC_MLR = Roc_Auc(Predict=predict,
                                  Target=y_test)
net_benefit_MLR = calculate_net_benefit_model(thresh_group, predict_pro, y_test)
print('MLR Validation set ROC-AUC:     %.3f'% AUC_MLR)

###################################### EM-En ######################################
Fpr_EM,Tpr_EM,AUC_EM = EM.Fpr_EM,EM.Tpr_EM,EM.AUC_EM
predict_pro = np.array(EM.predict_val_pro)
net_benefit_EM = calculate_net_benefit_model(thresh_group, predict_pro, EM.target_test)

###################################### DCA Preprocess #################################
net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
#Plot
fig, ax = plt.subplots()
### DCA Curve
ax.plot(thresh_group, net_benefit_SVM, color = '#81D8D0', label = 'SVM')
ax.plot(thresh_group, net_benefit_RF, color = '#008280FF', label = 'RandomForest')
ax.plot(thresh_group, net_benefit_MLR, color = '#CE959B', label = 'MLR')
ax.plot(thresh_group, net_benefit_XGBoost, color = '#13393E', label = 'XGBoost')
ax.plot(thresh_group, net_benefit_GBDT, color = '#800020', label = 'GBDT')
ax.plot(thresh_group, net_benefit_EM, color = '#BB0021FF', label = 'EM-En')
ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
#Figure Configuration
ax.set_xlim(0,1)
ax.set_ylim(net_benefit_XGBoost.min() - 0.15, net_benefit_XGBoost.max() + 0.15)
ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
ax.grid('major')
ax.spines['right'].set_color((0.8, 0.8, 0.8))
ax.spines['top'].set_color((0.8, 0.8, 0.8))
ax.legend(loc = 'upper right')
plt.savefig('figure/DCA.pdf')
plt.show()

###################################### ROC Curve ######################################
plt.figure(figsize=(22,20), dpi=80)
# ROC Curve
plt.plot(Fpr_SVM, Tpr_SVM, color='#81D8D0',
             lw=3, label='SVM (AUC = %0.3f)' % AUC_SVM)
plt.plot(Fpr_RF, Tpr_RF, color='#008280FF',
             lw=3, label='RandomForest (AUC = %0.3f)' % AUC_RF)
plt.plot(Fpr_MLR, Tpr_MLR, color='#CE959B',
             lw=3, label='MLR (AUC = %0.3f)' % AUC_MLR)
plt.plot(Fpr_XGBoost, Tpr_XGBoost, color='#13393E',
             lw=3, label='XGBoost (AUC = %0.3f)' % AUC_XGBoost)
plt.plot(Fpr_GBDT, Tpr_GBDT, color='#800020',
             lw=3, label='GBDT (AUC = %0.3f)' % AUC_GBDT)
plt.plot(Fpr_EM, Tpr_EM, color='#BB0021FF',
             lw=3, label='EM-En (AUC = %0.3f)' % AUC_EM)
plt.plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver Operating Characteristic Curve', fontsize=22)
plt.legend(loc="lower right", fontsize=18)
plt.savefig('figure/Models_ROC.pdf')
plt.show()
