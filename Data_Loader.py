import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

####################################### LR step1 data #######################################
### Em-2Class
radiomics_data = pd.read_csv('data/SCI_data_original.csv')
radiomics_target = pd.read_csv('data/SCI_target.csv')
radiomics_data['ASIA_4'] = radiomics_target['ASIA'].values
radiomics_data['ASIA_2'] = radiomics_target['ASIA_2'].values
radiomics_data = radiomics_data[radiomics_data['ASIA_4'] > 1]

radiomics_data = radiomics_data[radiomics_data['group'] == 'wu']
radiomics_data.pop('group')

target_4 = radiomics_data['ASIA_4']
target_2 = radiomics_data['ASIA_2']

new_x = pd.DataFrame()
new_x = pd.concat([new_x, radiomics_data])

new_x.pop('ASIA_2')
new_x.pop('ASIA_4')

normalize = StandardScaler()
normalize.fit(new_x)
standardized_train = normalize.transform(new_x)
standardized_train = pd.DataFrame(standardized_train)
standardized_train.columns = new_x.columns

le = LabelEncoder()
target_4 = le.fit_transform(target_4)
x_train_C2_,x_test_C2_,y_train_C2_, y_test_C2_ = train_test_split(standardized_train,
                                                                  target_2,
                                                                  test_size = 0.2,
                                                                  random_state = 3276)
if __name__ == '__main__':
    print('\n========= Group Em-2Class =========')
    print(pd.Series(target_2).value_counts())

####################################### step2 data #######################################
step2_total_data = pd.DataFrame()
step2_total_data = pd.concat([step2_total_data,
                              radiomics_data])
step2_total_data = step2_total_data[step2_total_data['ASIA_4'] > 2]

target_s2t = step2_total_data['ASIA_4']
target_s2t = le.fit_transform(target_s2t)
target_s2t_ensenmble = step2_total_data['ASIA_2']

normalize.fit(step2_total_data)
step2_total_data = normalize.transform(step2_total_data)
step2_total_data = pd.DataFrame(step2_total_data)
step2_total_data.columns = radiomics_data.columns
step2_total_data['ASIA_4'] = target_s2t
step2_total_data.pop('ASIA_2')

step2_total_data['step2_0'] = 0
step2_total_data.loc[step2_total_data['ASIA_4'] == 0, 'step2_0'] = 1
step2_total_data['step2_1'] = 0
step2_total_data.loc[step2_total_data['ASIA_4'] == 1, 'step2_1'] = 1
step2_total_data['step2_2'] = 0
step2_total_data.loc[step2_total_data['ASIA_4'] == 2, 'step2_2'] = 1

target_step_total = step2_total_data['step2_0'].values

x_train_total,x_test_total,y_train_total, y_test_total = train_test_split(step2_total_data,
                                                                          target_step_total,
                                                                          test_size = 0.2,
                                                                          random_state = 29)

###################### Em-C
x_train_0,x_test_0,y_train_0,y_test_0 = x_train_total,x_test_total,y_train_total, y_test_total
x_train_0 = x_train_0.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)
x_test_0 = x_test_0.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)

if __name__ == '__main__':
    print('\n========= Group Em-C =========')
    print(pd.Series(step2_total_data['step2_0'].values).value_counts())

###################### Em-B
x_train_1,x_test_1,y_train_1,y_test_1 =x_train_total,x_test_total, x_train_total['step2_1'].values, \
    x_test_total['step2_1'].values
x_train_1 = x_train_1.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)
x_test_1 = x_test_1.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)

if __name__ == '__main__':
    print('\n========= Group Em-B =========')
    print(pd.Series(step2_total_data['step2_1'].values).value_counts())

###################### Em-A
x_train_2,x_test_2,y_train_2,y_test_2 =x_train_total,x_test_total, x_train_total['step2_2'].values, \
    x_test_total['step2_2'].values
x_train_2 = x_train_2.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)
x_test_2 = x_test_2.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)
if __name__ == '__main__':
    print('\n========= Group Em-A =========')
    print(pd.Series(y_train_2).value_counts())

###################### Em-3Class
x_train_3,x_test_3,y_train_3,y_test_3 =x_train_total,x_test_total, x_train_total['ASIA_4'].values, \
    x_test_total['ASIA_4'].values
x_train_3 = x_train_3.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)
x_test_3 = x_test_3.drop(labels=['step2_0','step2_1','step2_2','ASIA_4'], axis=1)

if __name__ == '__main__':
    print('\n========= Group Em-3Class =========')
    print(pd.Series(y_test_3).value_counts())

####################################### Em-2Class data #######################################
x_train_insert_2Class = pd.concat([
        pd.DataFrame(), x_train_C2_
    ])
x_train_insert_2Class.loc[:,'ASIA_2'] = y_train_C2_.values
x_train_insert_2Class = x_train_insert_2Class.loc[x_train_insert_2Class['ASIA_2']==0,:]
x_train_insert_2Class.loc[:,'ASIA_4'] = 0

x_test_insert_2Class = pd.concat([
        pd.DataFrame(), x_test_C2_
    ])
x_test_insert_2Class.loc[:,'ASIA_2'] = y_test_C2_.values
x_test_insert_2Class = x_test_insert_2Class.loc[x_test_insert_2Class['ASIA_2']==0,:]
x_test_insert_2Class.loc[:,'ASIA_4'] = 0

x_train_insert_step2 = pd.concat([
        pd.DataFrame(), x_train_total
    ])
x_train_insert_step2.loc[:,'ASIA_2'] = 1
x_train_insert_step2.loc[:,'ASIA_4'] = x_train_total.loc[:,'ASIA_4'].values+1

x_test_insert_step2 = pd.concat([
        pd.DataFrame(), x_test_total
    ])
x_test_insert_step2.loc[:,'ASIA_2'] = 1
x_test_insert_step2.loc[:,'ASIA_4'] = x_test_total.loc[:,'ASIA_4'].values+1

x_train_C2 = pd.concat([
    x_train_insert_2Class,x_train_insert_step2
],axis=0, join='inner')
y_train_C2 = x_train_C2.loc[:,'ASIA_2'].values
y_train_ensemble = x_train_C2.loc[:,'ASIA_4'].values
x_train_C2 = x_train_C2.drop(['ASIA_2','ASIA_4'],axis=1)

x_test_C2 = pd.concat([
    x_test_insert_2Class,x_test_insert_step2
],axis=0, join='inner')
y_test_C2 = x_test_C2.loc[:,'ASIA_2'].values
y_test_ensemble = x_test_C2.loc[:,'ASIA_4'].values
x_test_C2 = x_test_C2.drop(['ASIA_2','ASIA_4'],axis=1)

x_total_C2 = pd.concat([
    x_train_C2, x_test_C2
], axis=0, join='inner')
y_total_C2 = np.concatenate([y_train_C2,y_test_C2])
