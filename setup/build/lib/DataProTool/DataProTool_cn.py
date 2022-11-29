"""
DataProTool
=====
简介
----

    此库是一个支持特征工程和数据处理高级工具的库。
    库由张杰郗独立开发.作者B站: https://space.bilibili.com/666767280?spm_id_from=333.788.0.0
    邮箱: zhangjiexi66696@outlook.com

依赖库
-----
    numpy ~= 1.23.3
    pandas ~= 1.5.0
    scikit-learn ~= 1.0.2
    tqdm ~=4.64.1
"""

# -*- coding: utf-8 -*-
# Author: Zhang Jiexi <zhangjiexi66696@outlook.com>


import numpy as np
import pandas as pd
import sklearn.base as base
import sklearn.feature_selection as feature_selection
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import tqdm
import gc
import copy

__version__ = '1.2.0'

__dependent__ = [["numpy","~=1.23.3"],["pandas","~=1.5.0"],["scikit-learn","~=1.0.2"],["tqdm","~=4.64.1"]]

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


def one_hot(data: pd.DataFrame,columns:list=[],drop: str='if_binary')-> pd.DataFrame:
    """
    对data进行OneHot编码,返回命名规范化的dataframe.

    Args:
        data(pd.DataFrame): 需要被OneHot编码的dataframe.
        columns(list): 需要被OneHot编码的columns.
        drop(str, optional): 是否取消对二分类的OneHot. 默认'if_binary'.
    
    Returns:
        data(pd.DataFrame):Onehot后的全部data(包括未OneHot的数据).
    """

    emc = preprocessing.OneHotEncoder(drop='if_binary') if drop == 'if_binary' else preprocessing.OneHotEncoder()
    if columns == []:
        columns = data.columns
        flag = False
    else:
        flag = True

    data_use = data[columns]
    columns_else = [i for i in data.columns if i not in columns]
    data_else = data[columns_else]
    result = emc.fit_transform(data_use).toarray()
    columns_new = one_hot_columns(columns,emc.categories_)

    dataframe = pd.DataFrame(data=result,columns=columns_new)
    if flag:
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
    此类用于实现以下特征衍生
        交叉组合特征衍生:
         cross_combination_feature_derivation()

        多项式特征衍生:
         polynomial_feature_derivation()

        四则运算特征衍生:
         four_arithmetic_feature_derivation()

        分组特征衍生: 
         group_freature_derivation()

        目标编码特征衍生:
         target_encode_derivation()
        
        时序特征衍生: 
         time_feature_dervation()

    二阶特征衍生建议
    ===============
        1.分组统计特征衍生 + 四则运算特征衍生
        -----------------------------------
            流量平滑特征: 分组统计特征衍生主要特征数据 / (各个衍生出特征的平均值 + ε), ε是为了防止除以0.
            黄金组合特征: 分组统计特征衍生主要特征数据 - 各个衍生出特征的平均值.
            gap: 一个特征的上四分位数 - 其下四分位数.
            数据偏度: 一个特征的平均值 - 其中位数 或 一个特征的平均值 / 其中位数
            变异系数: 一个特征的标准差 / (一个特征的均值 + ε), ε是为了防止除以0.
        
    """

    def cross_combination_feature_derivation(self,data:pd.DataFrame,columns:list=[],is_onehot:bool=True)->pd.DataFrame:
        """
        对数据进行双变量交叉组合特征衍生.

        Args
        ----
            data (pd.DataFrame): 需要进行处理的数据.
            columns (list):要进行特征衍生的列名,如果为 [] 则对所有列进行特征衍生.
            is_onehot (bool, optional): 是否对处理后的数据进行OneHot编码,默认为True.

        Returns
        -------
            data(pd.DataFrame): 经过交叉组合特征衍生后的数据.

        使用建议
        -----------------
           建议对离散型特征进行此特征衍生.
        """
        feature_new = []

        if columns == []:
            columns = data.columns
        features = data[columns]

        for x1_index, x1_feature in enumerate(features):
            for x2_index in range(x1_index+1,len(columns)): 
                new_name = x1_feature +"&" + columns[x2_index]
                new_feature = pd.DataFrame(data= data[x1_feature].astype("str") + "&" + data[columns[x2_index]].astype("str"),columns=[new_name])
                feature_new.append(new_feature)
        
        features_new = pd.concat(feature_new,axis=1)

        if is_onehot:
            features_new = one_hot(features_new)
            return pd.concat([data,features_new],axis=1)
        else:
            return pd.concat([data,features_new],axis=1)
    
    def polynomial_feature_derivation(self,data:pd.DataFrame,columns:list=[],degree:int=2,feature_interaction:int=0,include_bias:bool=False)->tuple[pd.DataFrame, preprocessing.PolynomialFeatures]:
        """
        对数据进行多项式特征衍生.

        Args
        ----
            data (pd.DataFrame): 需要进行多项式特征衍生的数据.
            columns (list): 要进行特征衍生的列名,如果为 [] 则对所有列进行特征衍生.
            degree (int, optional): 多项式最高次数,默认为2.
            feature_interaction (int, optional): 特征衍生形式. 0 表示不生成交互特征,
                1 表示只生成交互特征, 2表示生成交互特征与单个特征.默认为0.
            include_bias (bool, optional):是否包括偏差列,即该特征中所有多项式幂均为零.默认为False.

        Returns
        -------
            data(pd.DataFrame): 经过多项式特征衍生后的数据.
            model(sklearn.preprocessing.PolynomialFeatures):进行多项式特征衍生的模型.

        使用建议
        -------
            建议对连续性特征进行此特征衍生.
        """
        if feature_interaction not in [0,1,2]:
            raise ValueError("feature_interacation 只能为 0,1 或 2")
        
        feature_interaction = [0,True,False][feature_interaction]

        if columns != []:
            if feature_interaction == 0:
                data_p = pd.DataFrame()
                for each in columns:
                    derivator = preprocessing.PolynomialFeatures(degree=degree,include_bias=include_bias)
                    data_d = derivator.fit_transform(data[each])
                    data_d = pd.DataFrame(data=data_d, columns=derivator.get_feature_names_out())
                    pd.concat([data_p,data_d],axis=1)
            else:
                derivator = preprocessing.PolynomialFeatures(degree=degree,interaction_only=feature_interaction,include_bias=include_bias)
                data_p = derivator.fit_transform(data[columns])

            columns_new = derivator.get_feature_names_out()
            columns_old = data.columns
            columns_remain = list(set(columns_old) - set(columns))
            data_p = pd.DataFrame(data=data_p,columns=columns_new)
            data = pd.concat([ data[columns_remain], data_p ],axis=1)
            return data, derivator
        else:
            if feature_interaction == 0:
                data_p = pd.DataFrame()
                for each in data.columns:
                    derivator = preprocessing.PolynomialFeatures(degree=degree,include_bias=include_bias)
                    data_d = derivator.fit_transform(data[each])
                    data_d = pd.DataFrame(data=data_d, columns=derivator.get_feature_names_out())
                    pd.concat([data_p,data_d],axis=1)
            else:
                derivator = preprocessing.PolynomialFeatures(degree=degree,interaction_only=feature_interaction,include_bias=include_bias)
                data_p = derivator.fit_transform(data)

            columns_new = derivator.get_feature_names_out()
            data = pd.DataFrame(data=data_p,columns=columns_new)
            return data, derivator
    

    def four_arithmetic_feature_derivation(self,data:pd.DataFrame,columns:list=[],operations:list=["+","-","*","/"])->pd.DataFrame:
        """
        对数据进行四则运算特征衍生.

        Args
        ----
            data (pd.DataFrame): 需要进行多项式特征衍生的数据.
            columns (list): 要进行特征衍生的列名,如果为 [] 则对所有列进行特征衍生.
            operations (list, optional): 要进行的操作, 只能包括 '+ - * /'中的项. 默认为 ["+","-","*","/"].

        Returns
        -------
            data(pd.DataFrame): 经过四则运算特征衍生后的数据.
        """
        for each in operations:
            if each not in ["+","-","*","/"]:
                raise ValueError("operations 只能包含 '+ - * / ' ")

        if columns==[]:
            columns = data.columns

        new_features = [data]
        for pos,x1 in enumerate(columns):
            for x2 in columns[pos+1:]:
                if "+" in operations:
                    new_feature = pd.DataFrame(data=data[x1] + data[x2],columns=[x1+"+"+x2])
                    new_features.append(new_feature)
                if "*" in operations:
                    new_feature = pd.DataFrame(data=data[x1] * data[x2],columns=[x1+"*"+x2])
                    new_features.append(new_feature)

                if "-" in operations:
                    new_feature = pd.DataFrame(data=data[x1] - data[x2],columns=[x1+"-"+x2])
                    new_features.append(new_feature)
                    new_feature = pd.DataFrame(data=data[x2] - data[x1],columns=[x2+"-"+x1])
                    new_features.append(new_feature)
                if "/" in operations:
                    new_feature = pd.DataFrame(data=data[x1] / data[x2],columns=[x1+"/"+x2])
                    new_features.append(new_feature)
                    new_feature = pd.DataFrame(data=data[x2] / data[x1],columns=[x2+"/"+x1])
                    new_features.append(new_feature)

        return pd.concat(new_features,axis=1)

    def group_freature_derivation(self, data:pd.DataFrame, maincol:list,aggs:dict)-> pd.DataFrame:
        """
        对数据进行分组特征衍生.

        Args
        ----
            data (pd.DataFrame):需要进行分组特征衍生的数据.
            maincol (list): 主要特征名,只能有一个.
            aggs (list): agg函数的参数.

            参数aggs格式:
                {特征名(str) : [特征操作(str) , ... ] , ...}
            特征操作支持: max  min  mean  var  skew  median  count
            nunique  quantile-0.75(上四分位数)  quantile-0.25(下四分位数)
            ALL-num(常用统计量内的所有连续型变量的操作): ['max','min','mean','var','skew','median','quantile-0.25','quantile-0.75']
            ALL-cat(常用统计量内的所有离散型变量的操作): ['max','min','mean','var','median','quantile-0.25','quantile-0.75']
            e.g:
                {'Ages':['max', 'mean'], 'Plcass':['min','std']}

        Returns
        -------
            data(pd.DataFrame): 分组特征衍生后的全部数据,包括未进行特征工程的数据.

        Aggs参数解释
        -----------
            max,min:最值.
            mean/var:均值/方差.
            skew:数据分布偏度.小于零向左偏,大于零向右偏.
            median:中位数.
            quantile:2/4分位数.
        """
        def quantile75(x:pd.DataFrame):
            return x.quantile(0.75)
        
        def quantile25(x:pd.DataFrame):
            return x.quantile(0.25)

        columns = copy.deepcopy(maincol)
        mainstr = copy.deepcopy(columns[0])
        aggs = copy.deepcopy(aggs)

        for each in aggs.keys():
            if 'ALL-num' in aggs[each]:
                aggs[each] = ['max','min','mean','var','skew','median','quantile-0.25','quantile-0.75']
            if 'ALL-cat' in aggs[each]:
                aggs[each] = ['max','min','mean','var','median','quantile-0.25','quantile-0.75']
            
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

        gc.collect()
        return dataframe

    def target_encode_derivation(self,data:pd.DataFrame,maincol:list,label_aggs:dict,kfold_length:int=0)->pd.DataFrame:
        """
        对数据进行目标编码.

        Args
        ----
            data (pd.Dataframe): 需要进行目标编码的数据(包括标签).
            maincol (list): 主要特征名,只能有一个.
            label_aggs (dict): agg函数的参数,只能对标签进行处理,参数格式同group_freature_derivation() 函数.
            参数label_aggs格式:
                {特征名(str) : [特征操作(str) , ... ] , ...}
            特征操作支持: max  min  mean  var  skew  median  count
            nunique  quantile-0.75(上四分位数)  quantile-0.25(下四分位数)
            ALL-num(常用统计量内的所有连续型变量的操作): ['max','min','mean','var','skew','median','quantile-0.25','quantile-0.75']
            ALL-cat(常用统计量内的所有离散型变量的操作): ['max','min','mean','var','median','quantile-0.25','quantile-0.75']
            e.g:
                {'Ages':['max', 'mean'], 'Plcass':['min','std']}
            kfold_length (int, optional): 进行kflod交叉生成特征每组数据的量,如果为0,则不进行交叉生成特征.默认为0.

        Returns
        -------
            data(pd.DataFrame): 处理后的数据 
        
        Label_aggs参数解释
        -----------
            max,min:最值.
            mean/var:均值/方差.
            skew:数据分布偏度.小于零向左偏,大于零向右偏.
            median:中位数.
            quantile:2/4分位数.

        """
        if kfold_length < 0 or type(kfold_length) == float:
            raise ValueError("kflod_length 必须为自然数")
        
        if kfold_length == 0:
            feature_derivated = self.group_freature_derivation(data=data,maincol=maincol,aggs=label_aggs)
            return feature_derivated
        
        else:
            feature_derivated = []
            columns_old = data.columns.to_numpy().tolist()
            columns_new = self.group_freature_derivation(data=data[0:1], maincol=maincol, aggs=label_aggs).columns.to_numpy().tolist()
            columns_derivated = list(set(columns_new) - set(columns_old))

            for i in tqdm.tqdm(range(data.shape[0] // kfold_length)):
                data_release = data[i*kfold_length:(i+1)*kfold_length].copy(deep=True)
                data_train = data[0:kfold_length*i].copy(deep=True)
                data_train = pd.concat([data_train,data[(i+1)*kfold_length:]],axis=0).copy(deep=True)

                data_train = self.group_freature_derivation(data=data_train, maincol=maincol, aggs=label_aggs)

                for each in data_release[maincol[0]].unique().tolist():
                    data_release.loc[(data_release[maincol[0]]==each),columns_derivated] = data_train.loc[(data_train[maincol[0]]==each),columns_derivated][0:1].values
                
                feature_derivated.append(data_release)

            
            if data.shape[0] // kfold_length != data.shape[0] / kfold_length:
                data_release = data[(data.shape[0] // kfold_length)*kfold_length:].copy(deep=True)
                data_train = data[0:(data.shape[0] // kfold_length)*kfold_length]
                
                data_train = self.group_freature_derivation(data=data_train, maincol=maincol, aggs=label_aggs)

                for each in data_release[maincol[0]].unique().tolist():
                    data_release.loc[(data_release[maincol[0]]==each),columns_derivated] = data_train.loc[(data_train[maincol[0]]==each),columns_derivated][0:1].values
                
                feature_derivated.append(data_release)
            
            return pd.concat(feature_derivated,axis=0)


    def time_feature_derivation(self, time_data:pd.Series, time_stamp:dict=None, high_precision:bool=False)-> pd.DataFrame:
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

        if high_precision:
            features_new[col_name + '_hour'] = time_data.dt.hour
            features_new[col_name + '_minute'] = time_data.dt.minute
            features_new[col_name + '_second'] = time_data.dt.second

        features_new[col_name + '_quarter'] = time_data.dt.quarter
        #features_new[col_name + '_weekofyear'] = time_data.dt.week_of_year
        features_new[col_name + '_dayofweek'] = time_data.dt.dayofweek
        features_new[col_name + '_weekend'] = (features_new[col_name + '_dayofweek'] > 5).astype('int')

        if high_precision:
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

            if high_precision:
                features_new['time_diff_seconds'+'_' +time_stamp_name] = time_diff.dt.seconds
                features_new['time_diff_hours'+'_' +time_stamp_name] = time_diff.values.astype('timedelta64[ns]').astype('int')
                features_new['time_diff_minutes'+'_' +time_stamp_name] = time_diff.values.astype('timedelta64[ns]').astype('int')

        return features_new



class FeatureFilter:
    """
    此类用于实现以下特征筛选:
        方差分析: analysis_variance
        特征递归消除(RFE): analysis_RFE
    """

    def analysis_variance(self, data:pd.DataFrame, labels:pd.DataFrame)->tuple[pd.DataFrame,feature_selection.SelectKBest]|tuple[pd.DataFrame,feature_selection.SelectKBest,list]:
        """
        对数据进行方差分析(ANOVA),并返回筛选后的特征、筛选模型与p_values.
        

        Args
        ----
            data (pd.DataFrame|pd.Series|np.ndarray): 需要进行方差分析的特征数据.
            labels (pd.DataFrame|pd.Series|np.ndarray): 数据的标签.

        Returns
        -------
            data(pd.DataFrame): 方差分析后被筛选出的特征数据,顺序为p_values从小到大.
            selector(feature_selection.SelectKBest): 训练好的模型.
            p_values(list,optional): 所有特征的p_values.

        使用建议
        -------
            方差分析适用与离散型标签与连续性特征.

        """
        if data.shape[0] != len(labels):
            raise ValueError("输入数据与标签数量不匹配")

        features = data.columns
        selector = feature_selection.SelectKBest(feature_selection.f_classif, k=data.shape[1])
        selector.fit(data,labels)
        index = selector.get_support().tolist()
        p_values = selector.pvalues_[index].tolist()
        feature_selected = features[index].tolist()
        feature_selected.sort(key=lambda p_values:p_values[0])
        data = pd.DataFrame(data=data[feature_selected],columns=feature_selected)

        return data, selector, p_values


    def analysis_RFE(self, data:pd.DataFrame|pd.Series, labels:pd.DataFrame|pd.Series, estimator:base.BaseEstimator
    , params:dict|None=None, test_data:pd.DataFrame|None=None,test_labels: pd.DataFrame|None=None,n_jobs:int=1, high_precision:bool=False)->tuple[pd.DataFrame,list]:
        """
        对数据进行特征递归消除,并返回筛选后的特征及筛选模型.

        Args
        ----
            data (pd.DataFrame | pd.Series): 需要进行RFE特征筛选的特征数据.
            labels (pd.DataFrame | pd.Series): 数据标签.
            estimator (base.BaseEstimator): 用来进行特征筛选的监督学习估计器.
            n_jobs (int, optional):  进行特征筛选的CPU数量,注意: 此参数仅对GridSearchCV有效,默认为1.
            verbose (int, optional): 进行特征筛选时输出模型训练过程的详细程度,默认为0.
            param (dict, optional): 在特征筛选是每个训练模型的网格搜索的参数范围.默认值为”None"(不进行网格搜索调参).
            test_data (pd.DataFrame|None, optional): 在每一轮特征筛选之后,用于评估模型准确性的特征数据.如果为"None",则将使用训练数据进行评估.默认值为"None".
            test_labels (pd.DataFrame|None, optional): 用于在每一轮特征筛选之后评估模型准确性的数据标签.如果为"None".则将使用训练数据进行评估.默认为"None".
            high_precision(bool, optional): 是否进行高精度搜索,默认为 False.

        Returns
        -------
            data(pd.Dataframe): 筛选出的特征.
            scores (list): 模型分数, 格式为 [本轮被剔除的特征名, 模型分数].

        使用建议
        -------
            如果high_precision为True,那么RFE的准确率会提高,但是会占用更多资源,消耗更多时间.因此不建议在配置不好的电脑上使用.
        """
        if type(test_data) != type(test_labels):
            raise ValueError("test_data 与 test_labels 必须为同为None或pd.DataFrame")
        if data.index.shape != labels.index.shape:
            raise ValueError("训练数据必须与标签数量保持一致")
        if test_data.index.shape != test_labels.index.shape:
            raise ValueError("测试数据必须与测试标签数量保持一致")
        
        if type(data) == pd.Series:
            data = pd.DataFrame(data)
        if type(labels) == pd.Series:
            labels = pd.DataFrame(labels)

        if test_data == None:
            test_data = data.copy(deep=True)
            test_labels = labels.copy(deep=True)
        
        if high_precision:
            data_search = data.copy(deep=True)
            feature_select_list = []
            scores = []

            for i in tqdm.tqdm(range(data.shape[1]-1)):
                i = (data.shape[1]-1) - i
                model_search = model_selection.GridSearchCV(estimator=estimator, param_grid=params, n_jobs=n_jobs)
                model_search.fit(data_search,labels.values)

                selector = feature_selection.RFE(model_search.best_estimator_,n_features_to_select=i).fit(data_search,labels.values)
                
                feature_out = list(set(selector.feature_names_in_.tolist())- set(selector.get_feature_names_out().tolist()))[0] 
                scores.append([feature_out,selector.score(test_data[selector.feature_names_in_],test_labels.values)])
                feature_select_list.append(feature_out)
                data_search = data[selector.get_feature_names_out()]

            gc.collect()

            feature_select_list = feature_select_list[::-1]
            feature_select_list.append(selector.get_feature_names_out().tolist()[0])

            data = data[feature_select_list]

            return data, scores
            
        else:
            data_search = data.copy(deep=True)
            feature_select_list = []
            scores = []

            model_search = model_selection.GridSearchCV(estimator=estimator,param_grid=params,n_jobs=n_jobs)
            model_search.fit(data_search,labels.values)

            for i in tqdm.tqdm(range(data.shape[1]-1)):
                i = (data.shape[1]-1) - i

                selector = feature_selection.RFE(model_search.best_estimator_,n_features_to_select=i).fit(data_search,labels.values)
                
                feature_out = list(set(selector.feature_names_in_.tolist())- set(selector.get_feature_names_out().tolist()))[0] 
                scores.append([feature_out,selector.score(test_data[selector.feature_names_in_],test_labels.values)])
                feature_select_list.append(feature_out)
                data_search = data[selector.get_feature_names_out()]
            
            gc.collect()

            feature_select_list = feature_select_list[::-1]
            feature_select_list.append(selector.get_feature_names_out().tolist()[0])

            data = data[feature_select_list]

            return data,scores
