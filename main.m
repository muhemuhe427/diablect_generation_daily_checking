import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score,f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc

import warnings
warnings.simplefilter('ignore')
  
  
  
  
train = pd.read_csv('比赛训练集.csv',encoding='gbk')
test = pd.read_csv('比赛测试集.csv',encoding='gbk')
sub = pd.read_csv('提交示例.csv')
data = pd.concat([train,test]).reset_index(drop = True)


## -------对类别特征 One-Hot编码
data['糖尿病家族史'] = data['糖尿病家族史'].apply(
    lambda x:'叔叔或姑姑有一方患有糖尿病' if x=='叔叔或者姑姑有一方患有糖尿病' else x)
df = pd.get_dummies(data['糖尿病家族史']).astype('int')
data = pd.concat([data,df],axis = 1)

## -------对值为0的4个特征值替换为np.nan
for i in ['口服耐糖量测试','胰岛素释放实验','肱三头肌皮褶厚度','体重指数']:
#     data[i] = data[i].apply(lambda x:np.nan if x<=0 else x)
    data[i] = data[i].apply(lambda x:np.nan if x==0 else x)
      
      
      
      
train = data[data['患有糖尿病标识'].notnull()].reset_index(drop = True)
test = data[~data['患有糖尿病标识'].notnull()].reset_index(drop = True)
feas = [i  for i in train.columns.tolist() if i not in ['编号','糖尿病家族史','患有糖尿病标识',]]

x_train = train[feas]
y_train = train['患有糖尿病标识']
x_test = test[feas]


THR = 0.5 #f1阈值
folds = 7 
seed = 2021
def lgb_model(train_x, train_y, test_x):
    
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    f1_scores = []
    test_pre = []
    Feass = pd.DataFrame()

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('----------------------------------- {} -----------------------------------'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
  pd.Series([1 if i >0.4 else 0 for i in lgb_test]).value_counts()
  
          
sub['label'] = [1 if i >0.4 else 0 for i in lgb_test]
sub.to_csv('base_94456_407.csv',index =False)


        train_matrix = lgb.Dataset(trn_x, label=trn_y)
        valid_matrix = lgb.Dataset(val_x, label=val_y)
        fea = pd.DataFrame()

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'tree_learner':'serial',
                'metric': 'auc',
                'min_child_weight': 6,
                'num_leaves': 2 ** 6,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.01,
                'seed': 2021,
                'nthread': 28,
                'n_jobs':4,
                'silent': True,
                'verbose': -1,
            }

        model = lgb.train(params, train_matrix, 1000, valid_sets=[train_matrix, valid_matrix], 
                          categorical_feature =[] ,verbose_eval=200,early_stopping_rounds=200)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        test_pre.append(test_pred)
        fea['feas'] = train_x.columns.tolist()
        fea['sorce'] = model.feature_importance()
        Feass = pd.concat([Feass,fea],axis = 0)
        print(list(sorted(zip(train_x.columns.tolist(), model.feature_importance()), key=lambda x:x[1], reverse=True))[:30])

            
        train[valid_index] = val_pred
        cv_scores.append(roc_auc_score(val_y, val_pred))
        f1_scores.append(f1_score(val_y,[1 if i>THR else 0 for i in val_pred]))
        
        
        print(cv_scores)
        print(f1_scores)
    test = sum(test_pre) / folds
    print(f"s_scotrainre_list:  {cv_scores}")
    print(f"s_auc_score_mean:  {np.mean(cv_scores)}")
    print(f"s_f1_score_mean:  {np.mean(f1_scores)}")
    print(f"s_score_std:  {np.std(cv_scores)}")

    return train, test, Feass


lgb_train, lgb_test ,Feass= lgb_model(x_train, y_train, x_test)
