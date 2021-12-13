# !pip install pandas==0.24.2 --user
# !pip install lightgbm==2.3.1 --user
# !pip install xgboost==1.1.1 --user

# !cd ./model
import  os

print(os.getcwd())#获取当前工作目录路径
print(os.path.abspath('.')) #获取当前工作目录路径
# print os.path.abspath('test.txt') #获取当前目录文件下的工作目录路径
print(os.path.abspath('..')) #获取当前工作的父目录 ！注意是父目录路径
print(os.path.abspath(os.curdir))#获取当前工作目录路径

# coding: utf-8
import multiprocessing
from collections import Counter
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from sklearn.model_selection import KFold
import gc
from sklearn import preprocessing
from scipy.stats import entropy
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve
import datetime
import time
from itertools import product

nowtime = datetime.date.today()
nowtime = str(nowtime)[-5:]
print(nowtime)
warnings.filterwarnings('ignore')


# ==========
# Fzq's part
# ==========
def employmentLength_trans(x):
    if x == r'\N' or x == -999 or x == '-999':
        return -999
    elif x == '< 1 year':
        return 0.5
    elif x == '10+ years':
        return 12
    else:
        return int(x.split(' ')[0][0])


def earliesCreditLine_month_trans(x):
    x = x.split('-')[0]
    dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    return dict[x]


def grade_trans(x):
    dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    return dict[x]


def subGrade_trans(x):
    dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    return dict[x[0]] * 5 + int(x[1])


def myMode(x):
    # 众数
    return np.mean(pd.Series.mode(x))


def myRange(x):
    # 最大最小差值
    return pd.Series.max(x) - pd.Series.min(x)


def data_preprocess(DATA_PATH):
    train_label, train, test = load_dataset(DATA_PATH=DATA_PATH)

    # 拼接数据
    data = pd.concat([train, test], axis=0, ignore_index=True)
    print('train与test拼接后：', data.shape)

    n_feat = [f for f in data.columns if f[0] == 'n']

    name_list = ['max', 'sum', 'mean', 'median', 'skew', 'std']
    stat_list = ['max', 'sum', 'mean', 'median', 'skew', 'std']

    for i in range(len(name_list)):
        data['n_fea_{}'.format(name_list[i])] = data[n_feat].agg(stat_list[i], axis = 1)
    print('n特征处理后：', data.shape)

    # count编码，以count计数作为值
    count_list = ['subGrade', 'grade', 'postCode', 'regionCode', 'homeOwners', 'title','employmentTitle','employmentLength']
    data = count_coding(data, count_list)
    print('count编码后：', data.shape)

    # 选取和price相关性强的分类和数值特征进行一阶二阶交叉
    cross_cat = ['subGrade', 'grade', 'employmentLength', 'term', 'homeOwner', 'postCode', 'regionCode','employmentTitle','title']
    cross_num = ['dti', 'revolBal','revolUtil', 'ficoRangeHigh', 'interestRate', 'loanAmnt', 'installment', 'annualIncome', 'n14',
                 'n2', 'n6', 'n9', 'n5', 'n8']
    data = cross_cat_num(data, cross_num, cross_cat)  # 一阶交叉
    print('一阶特征处理后：', data.shape)
    data = cross_qua_cat_num(data)  # 二阶交叉
    print('二阶特征处理后：', data.shape)

    # 缺失值处理
    data[['employmentLength']].fillna(-999, inplace=True)
    for tmp in count_list:
        del data[tmp+'_count']
    cols = ['employmentTitle', 'employmentLength', 'postCode', 'dti', 'pubRecBankruptcies', 'revolUtil', 'title',
            'n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
    for col in cols:
        data[col].fillna(-999, inplace=True)
    print('缺失值情况：', data.isnull().sum())

    data['grade'] = data['grade'].apply(lambda x: grade_trans(x))
    data['subGrade'] = data['subGrade'].apply(lambda x: subGrade_trans(x))

    data['employmentLength'] = data['employmentLength'].apply(lambda x: employmentLength_trans(x))

    data['issueDate_year'] = data['issueDate'].apply(lambda x: int(x.split('-')[0]))
    data['issueDate_month'] = data['issueDate'].apply(lambda x: int(x.split('-')[1]))
    data['issueDate_day'] = data['issueDate'].apply(lambda x: transform_day(x))
    data['issueDate_week'] = data['issueDate_day'].apply(lambda x: int(x % 7) + 1)

    data['earliesCreditLine_year'] = data['earliesCreditLine'].apply(lambda x: 2020 - (int(x.split('-')[-1])))
    data['earliesCreditLine_month'] = data['earliesCreditLine'].apply(lambda x: earliesCreditLine_month_trans(x))
    data['earliesCreditLine_all_month'] = data['earliesCreditLine'].apply(lambda x: data['earliesCreditLine_year'] * 12 - data['earliesCreditLine_month'])

    del data['issueDate']
    del data['earliesCreditLine']

    print('预处理完毕：', data.shape)

    return data, train_label


