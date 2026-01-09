'''
Params of All Models
'''
params_xgboost = {
    'booster':'gbtree',
    'eta':0.05,
    'gamma':0,
    'max_depth':6,
    'max_delta_step':0,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'lambda':0.01,
    'alpha': 0,
    'objective': 'multi:softmax',
    'num_class':4,
    'base_score':0.1,
    'eval_metric':'auc',
    'seed':2020,
    'learning_rate':0.001,
    'tree_method':'gpu_hist'
}

params_prob = {
    'booster':'gbtree',
    'eta':0.05,
    'gamma':0,
    'max_depth':6,
    'max_delta_step':0,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda':0.01,
    'alpha': 0,
    'objective': 'multi:softprob',
    'num_class':4,
    'base_score':0,
    'eval_metric':'auc',
    'seed':2020,
    'tree_method':'gpu_hist'
}