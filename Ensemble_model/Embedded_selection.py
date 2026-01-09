from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import Data_Loader as data
from Metrics import MyMetrics

############################################# Feature Selection #############################################
datas = data.x_total_C2
target = data.y_total_C2

LR_ = LR(
    solver='liblinear',
    C=2.2,
    random_state=420,
    max_iter=1000,
    class_weight={0: 2.25, 1: 1}
    )

X_embedded = SelectFromModel(LR_, threshold=0.607).fit(datas, target)
feature_bull = X_embedded.get_support()
LR_.fit(datas.loc[:, feature_bull], target)

if __name__ == '__main__':
    try:
        sensitivity,specificity,precision_train,f1_score_train = MyMetrics(Predict=LR_.predict(datas.loc[:,feature_bull]),
                                                      Target=target,
                                                      twoClass=True)

        print(cross_val_score(LR_, datas.loc[:,feature_bull], target, cv=5).mean())
    except:
        print('Error!')

# pd.DataFrame(datas.loc[:,feature_bull]).to_csv('result/ASIA_2Classes_feature_new.csv')

###################### threshold
fullx = []
fsx = []

LR_ = LR(
    solver='liblinear',
    C=2,
    random_state=420,
    max_iter=100,
    class_weight={0: 2.4, 1: 1}
    )

threshold = np.linspace(0, abs(LR_.fit(datas, target).coef_).max(),20)
k = 0

for i in threshold:
    X_embedded = SelectFromModel(LR_, threshold=i).fit_transform(datas, target)
    fullx.append(cross_val_score(LR_, datas, target, cv=5).mean())
    fsx.append(cross_val_score(LR_, X_embedded, target, cv=5).mean())
    print((threshold[k], X_embedded.shape[1]))
    k += 1

plt.figure(figsize=(20,5))
plt.plot(threshold, fullx, label='full')
plt.plot(threshold, fsx, label='feature selection')
plt.xticks(threshold)
plt.legend()
plt.show()

###################### C
fullx = []
fsx = []

C = np.arange(0.1, 10, 0.1)
C1 = np.arange(0.51, 1.01, 0.005)

for i in C:
    LR_ = LR(
        solver='liblinear',
        C=i,
        random_state=420,
        max_iter=1000,
        class_weight={0: 2.4, 1: 1})
    fullx.append(cross_val_score(LR_, datas, target, cv=5).mean())
    X_embedded = SelectFromModel(LR_, threshold=0.6066820123941425).fit_transform(datas, target)
    fsx.append(cross_val_score(LR_, X_embedded, target, cv=5).mean())

print(max(fsx), C[fsx.index(max(fsx))])

plt.figure(figsize=(20,5))
plt.plot(C, fullx, label='full')
plt.plot(C, fsx, label='feature selection')
plt.xticks(C)
plt.legend()
plt.show()

