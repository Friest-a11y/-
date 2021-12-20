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

################################################1
#均值编码的实现，为了缓解 target encoding的偏差问题，出现了后来的mean encoding以及加入噪声的target encoding，mean encoding引入了类似于集成的思想，在不同的原始数据的抽样子集下计算target encoding值，然后平均。
#数学基础较为复杂，不做展开
#一个MeanEncoder对象可以提供fit_transform和transform方法，不支持fit方法，暂不支持训练时的sample_weight参数。
class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new

################################################2


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

# ==========
# gsdj's part
# ==========

### count编码
def count_coding(df, fea_col):
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return (df)


# 定义交叉特征统计
def cross_cat_num(df, num_col, cat_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
            })
            df = df.merge(feat, on=f1, how='left')
    return (df)


def cross_qua_cat_num(df):
    for f_pair in tqdm([
        ['subGrade', 'regionCode'], ['grade', 'regionCode'], ['subGrade', 'postCode'], ['grade', 'postCode'], ['employmentTitle','title'],
        ['regionCode','title'], ['postCode','title'], ['homeOwnership','title'], ['homeOwnership','employmentTitle'],['homeOwnership','employmentLength'],
        ['regionCode', 'postCode']
    ]):
        # 共现次数
        df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)['id'].transform('count')
        # n unique、熵
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[0], how='left')
        df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
            '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
            '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[1], how='left')
        # 比例偏好
        df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
        df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']
    return (df)


# count编码
def count_coding(df, fea_col):
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return (df)

def gen_basicFea(data):
    data['avg_income'] = data['annualIncome'] / data['employmentLength']
    data['total_income'] = data['annualIncome'] * data['employmentLength']
    data['avg_loanAmnt'] = data['loanAmnt'] / data['term']
    data['mean_interestRate'] = data['interestRate'] / data['term']
    data['all_installment'] = data['installment'] * data['term']

    data['rest_money_rate'] = data['avg_loanAmnt'] / (data['annualIncome'] + 0.1)  # 287个收入为0
    data['rest_money'] = data['annualIncome'] - data['avg_loanAmnt']

    data['closeAcc'] = data['totalAcc'] - data['openAcc']
    data['ficoRange_mean'] = (data['ficoRangeHigh'] + data['ficoRangeLow']) / 2
    del data['ficoRangeHigh'], data['ficoRangeLow']

    data['rest_pubRec'] = data['pubRec'] - data['pubRecBankruptcies']

    data['rest_Revol'] = data['loanAmnt'] - data['revolBal']

    data['dis_time'] = data['issueDate_year'] - (2020 - data['earliesCreditLine_year'])
    for col in ['employmentTitle', 'grade', 'subGrade', 'regionCode', 'issueDate_month', 'postCode']:
        data['{}_count'.format(col)] = data.groupby([col])['id'].transform('count')

    return data


def plotroc(train_y, train_pred, test_y, val_pred):
    lw = 2
    ##train
    fpr, tpr, thresholds = roc_curve(train_y.values, train_pred, pos_label=1.0)
    train_auc_value = roc_auc_score(train_y.values, train_pred)
    ##valid
    fpr, tpr, thresholds = roc_curve(test_y.values, val_pred, pos_label=1.0)
    valid_auc_value = roc_auc_score(test_y.values, val_pred)

    return train_auc_value, valid_auc_value
