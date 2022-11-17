"""
DataProTools
=====
简介
----
    此库为numpy、pandas数据处理拓展库,旨在方便日常数据处理.
    库由张杰郗<zhangjiexi66696@outlook.com>独立开发.作者B站: https://space.bilibili.com/666767280?spm_id_from=333.788.0.0
    邮箱: zhangjiexi66696@outlook.com

依赖库
-----
    numpy ~= 1.23.3
    pandas ~= 1.5.0
    scikit-learn ~= 1.0.2
"""

# -*- coding: utf-8 -*-
# Author: Zhang Jiexi <zhangjiexi66696@outlook.com>


import numpy as np
import pandas as pd
import sklearn.base as base
import sklearn.feature_selection as feature_selection
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing

__version__ = '1.0.1'

def one_hot_columns(col_used:list, col_categories:list)-> list:
    """
    把onehot编码形成的特征名称命名规范化.

    Args:
        col_used(list): 之前的columns.
        col_categories(list): 被OneHot编码器编码得到的columns.

    Returns:
        col_new(list): 命名规范化后的特征名称.
    """

    col_new=[]
    for i,j in enumerate(col_used):
        if len(col_categories[i]) == 2:
            col_new.append(j)
        else:
            for f in col_categories[i]:
                name = str(j) + '_' + str(f)
                col_new.append(name)
    return col_new


def one_hot(data: pd.DataFrame,columns : list,drop: str='if_binary')-> pd.DataFrame:
    """
    对data进行OneHot编码,返回命名规范化的dataframe.

    Args:
        data(pd.DataFrame): 需要被OneHot编码的dataframe.
        columns(list): 需要被OneHot编码的columns.
        drop(str, optional): 是否取消对二分类的OneHot. 默认'if_binary'.
    
    Returns:
        data(pd.DataFrame):Onehot后的d全部ata(包括未OneHot的数据).
    """

    emc = preprocessing.OneHotEncoder(drop='if_binary') if drop == 'if_binary' else preprocessing.OneHotEncoder()
    data_use = data[columns]
    columns_else = [i for i in data.columns if i not in columns]
    data_else = data[columns_else]
    result = emc.fit_transform(data_use).toarray()
    columns_new = one_hot_columns(columns,emc.categories_)

    dataframe = pd.DataFrame(data=result,columns=columns_new)
    dataframe = pd.concat([data_else,dataframe],axis=1)
    return dataframe


def nan_count(data:pd.DataFrame,columns:list=['ALL'])-> pd.DataFrame:
    """
    获取dataframe关于缺失值的信息(缺失值个数,缺失值占比).

    Args:
        data (pd.DataFrame): 需统计的dataframe.
        columns (list, optional):需统计的列.默认['ALL'].

    Returns:
        data(pd.Dataframe): 以dataframe形式返回统计信息.
    """

    columns = data.columns if columns == ['ALL'] else columns
    nan_count = []
    nan_percent = []

    for each in columns:
        nan = data[each].isnull().sum()
        nan_p = nan/(len(data[each]))
        nan_count.append(nan)
        nan_percent.append(nan_p)
    
    dataframe = pd.DataFrame(list(zip(nan_count,nan_percent)), columns=['nan count', 'nan percent'])
    dataframe.index = columns
    return dataframe.sort_values(['nan percent'],ascending=[False])



class FreatureDerivation:
    """
    此类用于实现以下特征衍生:
        多项式特征衍生: polynomial_feature_derivation
        分组特征衍生: group_freature_derivation
        时序特征衍生: time_feature_dervation
    """
    
    def polynomial_feature_derivation(self,data:pd.DataFrame,columns:list,degree:int=2,interaction_only:bool=False,include_bias:bool=False)->pd.DataFrame:
        """
        对数据进行多项式特征衍生.

        Args:
            data (pd.DataFrame): 原始数据.
            columns (list): 要进行特征衍生的列名,如果为 [] 则对所有列进行特征衍生.
            degree (int, optional): 多项式次数,默认为2.
            interaction_only (bool, optional): 是否仅生成交互特征,默认为False.
            include_bias (bool, optional):是否包括偏差列,即该特征中所有多项式幂均为零.默认为False.

        Returns:
            data(pd.DataFrame): 经过多项式特征衍生后的数据.
        """
        derivator = preprocessing.PolynomialFeatures(degree=degree,interaction_only=interaction_only,include_bias=include_bias)
        data_p = derivator.fit_transform(data[columns])
        columns_new = derivator.get_feature_names()

        if columns != []:
            columns_old = data.columns
            columns_left = list(set(columns_old) - set(columns))
            data = pd.concat([ data[columns_left], data_p ],axis=1)
            return data
        else:
            data = pd.DataFrame(data=data_p,columns=columns_new)
            return data


    def group_freature_derivation(self, data:pd.DataFrame, maincol:list,aggs:dict)-> pd.DataFrame:
        """
        对数据进行分组特征衍生.

        参数
        ----

        Args:
            data (pd.DataFrame):需要进行分组特征衍生的数据.
            maincol (list): 主要列名,只能有一个.
            aggs (list): agg函数的参数.

            参数aggs格式:
                {特征名(str) : [特征操作(str) , ... ] , ...}
            特征操作支持: max  min  mean  var  skew  median  count
            nunique  quantile-0.75(上四分位数)  quantile-0.25(下四分位数)
            ALL-num(常用统计量内的所有连续型变量的操作)  ALL-cat(常用统计量内的所有离散型变量的操作)
            e.g:
                {'Ages':['max', 'mean'], 'Plcass':['min','std']}

        Returns:
            data(pd.DataFrame): 分组特征衍生后的全部数据,包括未进行特征工程的数据.

        常用统计量：
        ----------

        连续型变量:
            max,min:最值.
            mean/var:均值/方差.
            skew:数据分布偏度.小于零向左偏,大于零向右偏.
            median:中位数.
            quantile:2/4分位数.
        分类型变量:
            max,min:最值.
            mean/var:均值/方差.
            median:中位数.
            count:个数统计.
            nunique:类别数.
            quantile:2/4分位数.
        """

        def quantile75(x):
            return x.quantile(0.75)
        
        def quantile25(x):
            return x.quantile(0.25)

        columns = maincol
        mainstr = columns[0]

        for each in aggs.keys():
            if 'ALL-num' in aggs[each]:
                aggs[each] = ['max','min','mean','var','skew','median','quantile-0.25','quantile-0.75']
            if 'ALL-cat' in aggs[each]:
                aggs[each] = ['max','min','mean','var','median','count','nunique','quantile-0.25','quantile-0.75']
            
            columns.extend([ mainstr + '_' + each + '_' + stat for stat in aggs[each]])

        for each in aggs.keys():
            if 'quantile-0.75' in aggs[each]:
                aggs[each].remove('quantile-0.75')
                aggs[each].append(quantile75)
            if 'quantile-0.25' in aggs[each]:
                aggs[each].remove('quantile-0.25')
                aggs[each].append(quantile25)

        freature_data = data.groupby([mainstr]).agg(aggs).reset_index()
        freature_data.columns = columns
        dataframe = pd.merge(data,freature_data,how='left',on=mainstr)

        del aggs
        return dataframe
    

    def time_feature_derivation(self, time_data:pd.Series, time_stamp:dict=None, precision_high:bool=False)-> pd.DataFrame:
        """
        对数据进行时序特征衍生.

        参数:
        ---

        Args:
            timeSeries (pd.Series):要进行时序特征衍生的时序字段,时间默认格式: "Y-H-D H:M:S".
            time_stamp (pd.time_stamp, optional): 手动输入关键时间节点的时间戳. 默认为 无(None).
            timeStame 参数格式:
            {'关键时间节点名称':['时间点']...} ,时间点默认格式:"Y-H-D H:M:S".

            precision_high (bool, optional): 是否进行高精度(时分秒)计算. 默认为False.

        Returns:
            data(pd.DataFrame): 时序特征衍生后的数据.

        时序特征衍生内容:
        ----------------
            时间点的年、月、日(时、分、秒)提取.
            关键时间点与当前时间点的差(月、日、(时、分、秒)).
        """

        features_new = pd.DataFrame()
        time_data = pd.to_datetime(time_data)
        col_name = time_data.name

        features_new[col_name + '_year'] = time_data.dt.year
        features_new[col_name + '_month'] = time_data.dt.month
        features_new[col_name + '_day'] = time_data.dt.day

        if precision_high:
            features_new[col_name + '_hour'] = time_data.dt.hour
            features_new[col_name + '_minute'] = time_data.dt.minute
            features_new[col_name + '_second'] = time_data.dt.second

        features_new[col_name + '_quarter'] = time_data.dt.quarter
        #features_new[col_name + '_weekofyear'] = time_data.dt.week_of_year
        features_new[col_name + '_dayofweek'] = time_data.dt.dayofweek
        features_new[col_name + '_weekend'] = (features_new[col_name + '_dayofweek'] > 5).astype('int')

        if precision_high:
            features_new['hour_section'] = (features_new[col_name + '_hour'] //8 )

        
        timestamp_col_name = []
        timestame_col = []

        if time_stamp !=None:
            timestamp_col_name = list(time_stamp.keys)
            timestame_col = [pd.time_stamp(x) for x in list[time_stamp.values]]

        time_max = time_data.max()
        time_min = time_data.min()
        timestame_col.extend([time_max,time_min])
        timestamp_col_name.extend(['time_max','time_min'])

        for time_stamp,time_stamp_name in zip(timestame_col,timestamp_col_name) :
            time_diff = time_data - time_stamp
            features_new['time_diff_days'+'_' +time_stamp_name] = time_diff.dt.days
            features_new['time_diff_months'+'_' +time_stamp_name] = np.round(time_diff.dt.days / 30).astype('int')

            if precision_high:
                features_new['time_diff_seconds'+'_' +time_stamp_name] = time_diff.dt.seconds
                features_new['time_diff_hours'+'_' +time_stamp_name] = time_diff.values.astype('timedelta64[ns]').astype('int')
                features_new['time_diff_minutes'+'_' +time_stamp_name] = time_diff.values.astype('timedelta64[ns]').astype('int')

        return features_new



class FeatureFilter:
    """
    此类用于实现以下特征筛选:
        方差分析: analysis_variance   (适用于用离散型标签筛选连续性特征)
        特征递归消除(RFE): analysis_RFE(CV)
    """

    def analysis_variance(self, data:pd.DataFrame, labels:pd.DataFrame
    , keep_num:int, threshold:float=0, return_p:bool=False
    )->tuple[pd.DataFrame,feature_selection.SelectKBest]|tuple[pd.DataFrame,feature_selection.SelectKBest,list]:
        """
        对数据进行方差分析,并返回筛选后的特征、筛选模型与p_values(可选).
        注: 方差分析适用于离散型标签筛选连续型特征

        Args:
            data (pd.DataFrame|pd.Series|np.ndarray): 需要进行方差分析的特征数据.
            labels (pd.DataFrame|pd.Series|np.ndarray): 数据的标签.
            keep_num(int, >0): 要保留的特征个数.
            threshold(int, optional): 附加阈值条件, 默认为0(无),如果输入一个整数,将会在
            筛选完特征之后再选出P_values小于threshold的特征.
            return_p(bool, optional): 是否返回所有特征的显著性水平,默认为False.

        Returns:
            data(pd.DataFrame): 方差分析后被筛选出的特征数据,顺序为p_values从小到大.
            selector(feature_selection.SelectKBest): 训练好的模型.
            p_values(list,optional): 所有特征的p_values.

        """
        if data.shape[0] != len(labels):
            raise ValueError("输入数据与标签不匹配！")
        if keep_num <= 0 :
            raise ValueError("keep_num 必须为正整数！")
        if threshold < 0 :
            raise ValueError("threshold 必须为正数！")
        
        features = data.columns
        selector = feature_selection.SelectKBest(feature_selection.f_classif, k=keep_num)
        selector.fit(data,labels)
        index = selector.get_support().tolist()
        p_values = selector.pvalues_[index].tolist()
        feature_selected = features[index].tolist()
        if threshold > 0:
            for each in range(len(feature_selected)):
                if p_values[each] > threshold:
                    feature_selected.remove(each)
        feature_selected.sort(key=lambda p_values:p_values[0])
        
        data = pd.DataFrame(data=data[feature_selected],columns=feature_selected)

        if return_p:
            return data, selector, p_values
        else:
            return data, selector

    def analysis_RFE(self, data:pd.DataFrame|pd.Series, labels:pd.DataFrame|pd.Series, estimator:base.BaseEstimator
    , keep_num:int,cv:int|model_selection.KFold|model_selection.StratifiedKFold=0, n_jobs:int=1, verbose:int=0
    )->tuple[pd.DataFrame,feature_selection.RFE|feature_selection.RFECV]:
        """
        对数据进行特征递归消除,并返回筛选后的特征及筛选模型.

        Args:
            data (pd.DataFrame | pd.Series): 需要进行RFE/RFECV特征筛选的特征数据.
            labels (pd.DataFrame | pd.Series): 数据标签.
            estimator (base.BaseEstimator): 用来进行特征筛选的监督学习估计器.
            keep_num (int): 保留(RFE)/最少保留(RFECV)的特征数量.
            cv (int, optional): 进行RFECV的交叉验证折数,如果为0,则进行RFE,默认为0.
            n_jobs (int, optional): 进行特征筛选的CPU数量,注意: 此参数仅对RFECV有效,默认为1.
            verbose (int, optional):进行特征筛选时输出模型训练过程的详细程度,默认为0.

        Returns:
            data(pd.Dataframe): 筛选出的特征.
            selector(feature_selection.RFE|feature_selection.RFECV): 训练好的模型.
        """
        if type(cv)==int and cv < 0:
            raise ValueError("cv 为int类型时,必须大于或等于0")
        if keep_num <= 0:
            raise ValueError("keep_num 必须为正整数")

        if type(data) == pd.Series:
            data = pd.DataFrame(data)
        
        if cv == 0:
            selector = feature_selection.RFE(estimator,n_features_to_select=keep_num,verbose=verbose)
        else:
            selector = feature_selection.RFECV(estimator,min_features_to_select=keep_num,cv=cv,n_jobs=n_jobs,verbose=verbose)
        
        selector.fit_transform(data,labels)

        results = selector.support_.tolist()
        columns = data.columns[results]
        ranking = selector.ranking_.tolist()[0:len(columns)]
        ranking = [ [ranking[i],i] for i in range(len(ranking))]
        ranking.sort(key = lambda ranking: ranking[0])

        data_p = pd.DataFrame(data=data[columns],columns=columns)
        data = pd.DataFrame()
        for each in ranking:
            data[columns[each[1]]] = data_p[columns[each[1]]]
        
        return data, selector