from sklearn.metrics import confusion_matrix,auc,roc_curve
from sklearn import preprocessing
import numpy as np

def MyMetrics(Predict, Target, twoClass=True, Class=3):
    confusion_matrix_data = confusion_matrix(Target,Predict)
    smooth = 1e-5
    if twoClass:
        TP = confusion_matrix_data[0][0]
        FP = confusion_matrix_data[1][0]
        TN = confusion_matrix_data[1][1]
        FN = confusion_matrix_data[0][1]
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = (TP + smooth) / (TP + FN + smooth)
        # Specificity or true negative rate
        TNR = (TN + smooth) / (TN + FP + smooth)
        # Precision or positive predictive value
        PPV = (TP + smooth) / (TP + FP + smooth)
        # Fall out or false positive rate
        FPR = (FP + smooth) / (FP + TN + smooth)
    else:
        cm = confusion_matrix(Target,Predict, labels=range(Class))
        cm = cm.astype(np.float32)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = (TP + smooth) / (TP + FN + smooth)
        TPR = np.sum(TPR).tolist() / Class
        # Specificity or true negative rate
        TNR = (TN + smooth) / (TN + FP + smooth)
        TNR = np.sum(TNR).tolist() / Class
        # Precision or positive predictive value
        PPV = (TP + smooth) / (TP + FP + smooth)
        PPV = np.sum(PPV).tolist() / Class
        # Fall out or false positive rate
        FPR = (FP + smooth) / (FP + TN + smooth)
        FPR = np.sum(FPR).tolist() / Class

    F1 = (2 * PPV * TPR) / (PPV + TPR)
    return TPR,TNR,PPV,F1

def Roc_Auc(Predict, Target):
    lb = preprocessing.LabelBinarizer()
    y_onehot = lb.fit_transform(Target)
    pre_onehot = lb.fit_transform(Predict)
    FPR, TPR, threshold_train = roc_curve(y_onehot.ravel(), pre_onehot.ravel())
    auc_roc = auc(FPR, TPR)
    return FPR,TPR,auc_roc

def calculate_net_benefit_model(thresh_group, Predict_pro, Target):
    net_benefit_model = np.array([])
    lb = preprocessing.LabelBinarizer()
    y_onehot = lb.fit_transform(Target).ravel()
    y_pred_score = Predict_pro.ravel()
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_onehot, y_pred_label).ravel()
        n = len(y_onehot)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, Target):
    lb = preprocessing.LabelBinarizer()
    y_onehot = lb.fit_transform(Target).ravel()
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_onehot, y_onehot).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)
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

    return ax
