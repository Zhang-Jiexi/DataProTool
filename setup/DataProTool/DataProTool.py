"""
DataProTool
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
import sklearn.preprocessing as preprocessing
import sklearn.feature_selection as feature_selection

__version__ = '1.0.0'

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
        data(pd.DataFrame):Onehot后的d全部ata(包括未OneHot的数据)
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
        data (pd.DataFrame): 需统计的dataframe
        columns (list, optional):需统计的列.默认['ALL'].

    Returns:
        data(pd.Dataframe): 以dataframe形式返回统计信息
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
        分组特征衍生: group_freature_derivation
        时序特征衍生: time_feature_dervation
    """

    def group_freature_derivation(self, data:pd.DataFrame, maincol:list,aggs:dict)-> pd.DataFrame:
        """
        对数据进行分组特征衍生.

        参数：
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
        num_col = ['max','min','mean','var','skew','median','quantile-0.25','quantile-0.75']
        cat_col = ['max','min','mean','var','median','count','nunique','quantile-0.25','quantile-0.75']

        for each in aggs.keys():
            if 'ALL-num' in aggs[each]:
                aggs[each] = num_col
            if 'ALL-cat' in aggs[each]:
                aggs[each] = cat_col

            columns.extend([ mainstr + '_' + each + '_' + stat for stat in aggs[each]])

            if 'quantile-0.75' in aggs[each]:
                aggs[each].remove('quantile-0.75')
                aggs[each].append(quantile75)
            if 'quantile-0.25' in aggs[each]:
                aggs[each].remove('quantile-0.25')
                aggs[each].append(quantile25)

        freature_data = data.groupby([mainstr]).agg(aggs).reset_index()
        freature_data.columns = columns
        dataframe = pd.merge(data,freature_data,how='left',on=mainstr)

        return dataframe
    

    def time_feature_derivation(self, timeSeries:pd.Series, timeStamp:dict=None, precision_high:bool=False)-> pd.DataFrame:
        """
        对数据进行时序特征衍生

        参数:
        ---

        Args:
            timeSeries (pd.Series):要进行时序特征衍生的时序字段,时间默认格式: "Y-H-D H:M:S"
            timeStamp (pd.Timestamp, optional): 手动输入关键时间节点的时间戳. 默认为 无(None)
            timeStame 参数格式:
            {'关键时间节点名称':['时间点']...} ,时间点默认格式:"Y-H-D H:M:S"

            precision_high (bool, optional): 是否进行高精度(时分秒)计算. 默认为False.

        Returns:
            pd.DataFrame: 时序特征衍生后的数据

        时序特征衍生内容:
            时间点的年、月、日(时、分、秒)提取
            关键时间点与当前时间点的差(月、日、(时、分、秒))
        ----------------


        """

        features_new = pd.DataFrame()
        time_data = pd.to_datetime(timeSeries)
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

        if timeStamp !=None:
            timestamp_col_name = list(timeStamp.keys)
            timestame_col = [pd.Timestamp(x) for x in list[timeStamp.values]]

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
    """

    def analysis_variance(self, data:pd.DataFrame|pd.Series, labels:list, keep_num:int, threshold:float=0, return_p:bool=False)->pd.DataFrame|tuple[pd.DataFrame,list]:
        """
        对数据进行方差分析,并按照返回筛选过后指定个数的特征。
        注: 方差分析适用于离散型标签筛选连续型特征
        Args:
            data (pd.DataFrame | pd.Series): 需要进行方差分析的特征数据
            labels (list): 数据的标签
            keep_num(int, >0): 要保留的特征个数
            threshold(int, optional): 附加阈值条件, 默认为0(无),如果输入一个整数,将会在
            筛选完特征之后再选出P_values小于threshold的特征
            return_p(bool, optional): 是否返回所有特征的显著性水平,默认为False
        Returns:
            pd.DataFrame: 方差分析后被筛选出的特征数据

        """
        if data.shape[0] != len(labels):
            raise ValueError("输入数据与标签不匹配！")
        if keep_num <= 0 :
            raise ValueError("keep_num 必须为正整数！")
        if threshold < 0 :
            raise ValueError("threshold 必须为正数！")
        
        features = data.columns.tolist
        chooser = feature_selection.SelectKBest(feature_selection.f_classif, k=keep_num)
        chooser.fit(data,labels)
        index = chooser.get_support().tolist
        p_values = chooser.pvalues_.tolist[index]
        feature_selected = features[index]
        if threshold > 0:
            for each in p_values:
                if each > threshold:
                    feature_selected.remove(each)
        
        data_return = pd.DataFrame(data=data[feature_selected],columns=feature_selected)
        if return_p:
            return data_return, p_values
        else:
            return data_return
