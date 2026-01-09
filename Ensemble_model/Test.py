import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import Data_Loader as data
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import shap
from Metrics import *
import Logistic_Regression as LDR
import Em_En as Em_En
import Em_A as Em_A

####### load features
imp_s1 = LDR.imp_2class
imp_s2_0 = LDR.imp_0
imp_s2_1 = LDR.imp_1
imp_s2_2 = Em_A.imp_2

####### load train|test dataset
target_test = data.y_test_ensemble
target_train = data.y_train_ensemble

predict_train, predict_train_pro = Em_En.Em_en(train_data=data.x_train_C2,
                                               train_target=data.y_train_C2,
                                               val_data=data.x_test_C2,
                                               val_target=data.y_test_C2,
                                               train_mode=True)
predict_val, predict_val_pro = Em_En.Em_en(train_data=data.x_train_C2,
                                           train_target=data.y_train_C2,
                                           val_data=data.x_test_C2,
                                           val_target=data.y_test_C2,
                                           train_mode=False)

sensitivity,specificity,precision,f1_score = MyMetrics(Predict=predict_val,
                                              Target=target_test,
                                              twoClass=False,
                                              Class=4)
Fpr_EM,Tpr_EM,AUC_EM = Roc_Auc(Predict=predict_val,
                               Target=target_test)

##################### ROC and PR curve #####################
######### AUC
lb = preprocessing.LabelBinarizer()
y_onehot_train = lb.fit_transform(target_train)
pre_onehot_train = lb.fit_transform(predict_train)
fpr_train, tpr_train, threshold_train = roc_curve(y_onehot_train.ravel(), pre_onehot_train.ravel())
roc_auc_train = auc(fpr_train, tpr_train)

y_onehot_test = lb.fit_transform(target_test)
pre_onehot_test = lb.fit_transform(predict_val)
fpr_test, tpr_test, threshold_test = roc_curve(y_onehot_test.ravel(), pre_onehot_test.ravel())
roc_auc_test = auc(fpr_test, tpr_test)

##################### precision & recall #####################
precision_train, recall_train, threshold_PR_train = precision_recall_curve(y_onehot_train.ravel(),
                                                                           pre_onehot_train.ravel())

precision_test, recall_test, threshold_PR_test = precision_recall_curve(y_onehot_test.ravel(),
                                                                        pre_onehot_test.ravel())

##################### Confusion matrix #####################
confusion_matrix_train = confusion_matrix(target_train, predict_train)
confusion_matrix_test = confusion_matrix(target_test, predict_val)

#################### Plot ROC curve, PR curve and Confusion matrix heatmap #####################
if __name__ == '__main__':
    classes = ['D','C','B','A']
    plt.figure(figsize=(22,20), dpi=80)
    # ROC Curve
    plt.subplot(2,2,1)
    plt.plot(fpr_train, tpr_train, color='#008280FF',
                 lw=3, label='Training Set (AUC = %0.3f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='#BB0021FF',
                 lw=3, label='Validation Set (AUC = %0.3f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Receiver Operating Characteristic Curve', fontsize=22)
    plt.legend(loc="lower right", fontsize=18)
    # PR Curve
    plt.subplot(2,2,2)
    plt.plot(precision_train, recall_train, color='#008280FF', lw=3,
             label = 'Training Set')
    plt.plot(precision_test, recall_test, color='#BB0021FF', lw=3,
             label = 'Validation Set')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Precision Scores', fontsize=20)
    plt.ylabel('Recall Scores', fontsize=20)
    plt.title('Precision-Recall Curve', fontsize=22)
    plt.legend(loc="lower left", fontsize=18)
    # Training Confusion Matrix
    plt.subplot(2,2,3)
    plt.imshow(confusion_matrix_train, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Training Set Confusion Matrix', fontsize=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    thresh = confusion_matrix_train.max() / 2.
    iters_train = np.reshape([[[i,j] for j in range(4)] for i in range(4)],(confusion_matrix_train.size,2))
    for i, j in iters_train:
        plt.text(j, i, format(confusion_matrix_train[i, j]), fontsize = 20)
    plt.ylabel('Truth', fontsize = 20)
    plt.xlabel('Prediction', fontsize = 20)
    plt.tight_layout()
    # Validation Confusion Matrix
    plt.subplot(2,2,4)
    plt.imshow(confusion_matrix_test, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Validation Set Confusion Matrix', fontsize=22)
    plt.colorbar()
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    iters_test = np.reshape([[[i,j] for j in range(4)] for i in range(4)],(confusion_matrix_test.size,2))
    for i, j in iters_test:
        plt.text(j, i, format(confusion_matrix_test[i, j]), fontsize = 20)
    plt.ylabel('Truth', fontsize = 20)
    plt.xlabel('Prediction', fontsize = 20)
    plt.tight_layout()
    plt.savefig('figure/ASIA_model.pdf')
    plt.show()

#################### Plot Decision Boundaries #####################
###### t-SNE #######
t_sne = TSNE(n_components=2, random_state=10)
data_2class = data.standardized_train[imp_s1.columns[1:]]
data_2class = t_sne.fit_transform(data_2class)
x_train_2class,x_test_2class,y_train_2class, y_test_2class = train_test_split(
                                                                     data_2class,
                                                                     data.target_2,
                                                                     test_size = 0.2,
                                                                     random_state = 888)

################################### 2Class Loop
lda_3 = LDA(n_components=2)
data_pre = data.step2_total_data
data_pre = data_pre.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)
target_3class = data.target_s2t

data_3class = Em_En.Ensemble_Dataloder(data_pre)
lda_3.fit(data_3class,target_3class)
data_3class = lda_3.transform(data_3class)

x_train_3class, x_test_3class, y_train_3class, y_test_3class = train_test_split(
        data_3class,
        target_3class,
        test_size=0.2,
        random_state=29)


##### Decision Boundaries function #######
##### 2-Class Decision Boundaries #######
def plot_decision_boundary_2(model, X_train ,y_train, X_val ,y_val):
    # Training Set
    x_min_train, x_max_train = X_train[:,0].min() - 1, X_train[:,0].max() + 1
    y_min_train, y_max_train = X_train[:,1].min() - 1, X_train[:,1].max() + 1
    h = 1000
    xx_train, yy_train = np.meshgrid(np.linspace(x_min_train, x_max_train, h).reshape(-1,1),
                         np.linspace(y_min_train, y_max_train, h).reshape(-1,1))
    new_x_train = np.c_[xx_train.ravel(), yy_train.ravel()]
    Z_train = model.predict(new_x_train)
    Z_train = Z_train.reshape(xx_train.shape)
    # Validation Set
    x_min_val, x_max_val = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
    y_min_val, y_max_val = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
    xx_val, yy_val = np.meshgrid(np.linspace(x_min_val, x_max_val, h).reshape(-1, 1),
                                 np.linspace(y_min_val, y_max_val, h).reshape(-1, 1))
    new_x_val = np.c_[xx_val.ravel(), yy_val.ravel()]
    Z_val = model.predict(new_x_val)
    Z_val = Z_val.reshape(xx_val.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap_train = ListedColormap(['#008B4533', '#63187933'])
    # Plot
    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0
    plt.figure(figsize=(22, 10), dpi=80)
    plt.subplot(1,2,1)
    plt.contourf(xx_train, yy_train, Z_train, cmap=custom_cmap_train)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color = '#BB002199',
                marker='o', label='ASIA Level-D')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='darkorange',
                marker='x', label='ASIA Level-A,B and C')
    plt.title('Training Set', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(1, 2, 2)
    plt.contourf(xx_val, yy_val, Z_val, cmap=custom_cmap_train)
    plt.scatter(X_val[y_val == 0, 0], X_val[y_val == 0, 1], color='#BB002199',
                marker='o', label='ASIA Level-D')
    plt.scatter(X_val[y_val == 1, 0], X_val[y_val == 1, 1], color='darkorange',
                marker='x', label='ASIA Level-A,B and C')
    plt.title('Validation Set', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3,
                              borderaxespad=num4, fontsize=16)
    plt.savefig('figure/Dicision_boundary_2class_combine.pdf')
    plt.show()

def PolynomialLogisticsRegression_2(degree=1, C=0.5, penalty='none'):
  return Pipeline([
    ("poly", PolynomialFeatures(degree=degree)),
    ("std_scaler", StandardScaler()),
    ("log_reg", LogisticRegression(
                                   C=C,
                                   max_iter=50000,
                                   random_state=0))
  ])
lr_2class = PolynomialLogisticsRegression_2(degree=9, C=1)
lr_2class.fit(x_train_2class, y_train_2class)

if __name__ == '__main__':
    predict_2class = lr_2class.predict(x_test_2class)
    predict_2class_train = lr_2class.predict(x_train_2class)
    print('\n============ 2Class ============')
    print('Train Accuracy {:.3f}'.format(accuracy_score(predict_2class_train, y_train_2class)))
    print('Val   Accuracy {:.3f}'.format(accuracy_score(predict_2class, y_test_2class)))
    plot_decision_boundary_2(lr_2class, x_train_2class, y_train_2class, x_test_2class, y_test_2class)

###### 3-Class Decision Boundaries #######
def plot_decision_boundary_3(model, X_train ,y_train, X_val ,y_val):
    # Training Set
    x_min_train, x_max_train = X_train[:,0].min() - 1, X_train[:,0].max() + 1
    y_min_train, y_max_train = X_train[:,1].min() - 1, X_train[:,1].max() + 1
    h = 1000
    xx_train, yy_train = np.meshgrid(np.linspace(x_min_train, x_max_train, h).reshape(-1,1),
                         np.linspace(y_min_train, y_max_train, h).reshape(-1,1))
    new_x_train = np.c_[xx_train.ravel(), yy_train.ravel()]
    Z_train = model.predict(new_x_train)
    Z_train = Z_train.reshape(xx_train.shape)
    # Validation Set
    x_min_val, x_max_val = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
    y_min_val, y_max_val = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
    xx_val, yy_val = np.meshgrid(np.linspace(x_min_val, x_max_val, h).reshape(-1, 1),
                                 np.linspace(y_min_val, y_max_val, h).reshape(-1, 1))
    new_x_val = np.c_[xx_val.ravel(), yy_val.ravel()]
    Z_val = model.predict(new_x_val)
    Z_val = Z_val.reshape(xx_val.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap_train = ListedColormap(['#008B4533', '#63187933', '#3B499233'])
    # Plot
    num1 = 1.05
    num2 = 0
    num3 = 3
    num4 = 0
    plt.figure(figsize=(22, 10), dpi=80)
    plt.subplot(1,2,1)
    plt.contourf(xx_train, yy_train, Z_train, cmap=custom_cmap_train)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color = '#BB002199',
                marker='o', label='ASIA Level-C')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='darkorange',
                marker='x', label='ASIA Level-B')
    plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='#3B4992FF',
                marker='^', label='ASIA Level-A')
    plt.title('Training Set', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(1, 2, 2)
    plt.contourf(xx_val, yy_val, Z_val, cmap=custom_cmap_train)
    plt.scatter(X_val[y_val == 0, 0], X_val[y_val == 0, 1], color='#BB002199',
                marker='o', label='ASIA Level-C')
    plt.scatter(X_val[y_val == 1, 0], X_val[y_val == 1, 1], color='darkorange',
                marker='x', label='ASIA Level-B')
    plt.scatter(X_val[y_val == 2, 0], X_val[y_val == 2, 1], color='#3B4992FF',
                marker='^', label='ASIA Level-A')
    plt.title('Validation Set', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3,
               borderaxespad=num4, fontsize=16)
    plt.savefig('figure/Dicision_boundary_3class_combine.pdf')
    plt.show()

def PolynomialLogisticsRegression_3(degree=1, C=10, penalty='none'):
  return Pipeline([
    ("RF_entropy", RandomForestClassifier(n_estimators=1,
                              n_jobs=-1,
                              criterion='entropy',
                              class_weight={0:1.1, 1:1.3, 2:1},
                              random_state=6
                              ))
  ])
if __name__ == '__main__':
    lr_3class = PolynomialLogisticsRegression_3()
    lr_3class.fit(x_train_3class, y_train_3class)
    predict_3class = lr_3class.predict(x_test_3class)
    predict_3class_train = lr_3class.predict(x_train_3class)
    print('\n============ 3Class ============')
    print('Train Accuracy {:.3f}'.format(accuracy_score(predict_3class_train, y_train_3class)))
    print('Val   Accuracy {:.3f}'.format(accuracy_score(predict_3class, y_test_3class)))
    plot_decision_boundary_3(lr_3class, x_train_3class, y_train_3class, x_test_3class, y_test_3class)

##### SHAP values #######
# Create the explainer
feature_ensemble = pd.concat([pd.DataFrame(), data.x_train_C2])[imp_s1.columns[1:]]
feature_test = pd.concat([pd.DataFrame(), data.x_test_C2])[imp_s1.columns[1:]]
explainer = shap.Explainer(LDR.LR_1.predict, feature_ensemble)

# shap values
shap_values = explainer(feature_ensemble)

# Figures plot
if __name__ == '__main__':
    plt.figure()
    shap.summary_plot(shap_values, feature_ensemble,
                      class_names=['ASIA-low', 'ASIA-high'],
                      show=False,
                      # max_display=10,
                      plot_type='dot')
    plt.savefig('figure/Summary_plot_dot.pdf')
    plt.show()
    plt.close()


    shap.plots.heatmap(shap_values,
                       show=False,
                       max_display=20)
    plt.savefig('figure/Heatmap_plot.pdf')
    plt.show()
    plt.close()
