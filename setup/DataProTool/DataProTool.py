"""
DataProTool
=====

introduction
----

    It is a library that support advance tools in feature engineering and data progress.
    This library is independently developed by Zhang Jiexi. Author's e-mail: zhangjiexi66696@outlook.com

dependent libraries
-----

    numpy ~= 1.23.3
    pandas ~= 1.5.0
    scikit-learn ~= 1.0.2
    tqdm ~= 4.64.1
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

import DataProTool_cn as cn

__version__ = '1.2.0'

__dependent__ = [["numpy","~=1.23.3"],["pandas","~=1.5.0"],["scikit-learn","~=1.0.2"],["tqdm","~=4.64.1"]]

def one_hot_columns(col_used:list, col_categories:list)-> list:
    """
    Normalize the feature names after OneHot encoding.

    Args
    ----
        col_used(list): Columns before encoding by OneHot.
        col_categories(list): Columns which are after encoding by OneHot.

    Returns
    -------
        col_new(list): the feature names after OneHot encoding.
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
     Encode the data by OneHot and name the normalized data.

    Args
    ----
        data(pd.DataFrame):  Data that need to be encoded by OneHot.
        columns(list): Columns that need to be encoded by OneHot in the data.Defauts to [](OneHot all columns).
        drop(str, optional): Whether to OneHot for the secondary classification feature Default 'if_ binary'.
    
    Returns
    -------
        data(pd.DataFrame) Data after OneHot encoding,include the data which is not encode by OneHot.
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
    Get the missing values of the dataframe (number of missing values, percentage of missing values).

    Args
    ----
        data (pd.DataFrame): Dataframe to be counted.
        columns (list, optional):Columns to be counted in the data. Defaults to ['ALL'].

    Returns
    -------
        data(pd.Dataframe): Missing values informations of the data.
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
        This class is used to following feature derivate:

            cross combination feature derivation:
             cross_combination_feature_derivation()

            polynomial feature derivation:
             polynomial_feature_derivation()

            four arithmetic feature derivation:
             four_arithmetic_feature_derivation()

            group freature derivation: 
             group_freature_derivation()
            
            target encode derivation:
             target_encode_derivation()

            time feature dervation: 
             time_feature_dervation()

    Second order feature derivation suggestion
    =========================================
        1. group freature derivation + four arithmetic feature derivation
        -----------------------------------------------------------------
            Traffic smoothing feature: main features' data derived from grouping statistical characteristics/(average value of each derived features+ ε)
                ε To prevent dividing by 0
            Gold portfolio feature: Group statistical features derive main feature's data - average value of each derived features
            Gap: The upper quartile of a feature - its lower quartile
            Data skewness: average median of a feature or average/median of a feature
            Coefficient of variation: standard deviation of a feature/(mean value of a feature+ ε), ε To prevent dividing by 0
    """

    def cross_combination_feature_derivation(self,data:pd.DataFrame,columns:list=[],is_onehot:bool=True)->pd.DataFrame:
        """
        Derive data by bivariate cross combination feature derivation.

        Args
        ----
            data (pd.DataFrame): Data need to derived.
            columns (list): Features' names that need to derived in the data.
                If columns equals to [], the function will derive all the columns in the data.Defaults to [].
            is_onehot (bool, optional): Whether to onehot the data after derivation. Defaults to True.

        Returns
        -------
            data(pd.DataFrame): The data after deriving.

        Usage suggestions
        -----------------
            This function is usually suggested to use for discrete features.
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
        Derive data by polynomial feature derivation.

        Args
        ----
            data (pd.DataFrame): Data need to derived.
            columns (list): Features' names that need to derived in the data.
            If columns equals to [], the function will derive all the columns in the data.Defaults to [].
            degree (int, optional): The polynomial's highest degree.Defaults to 2.
            feature_interaction (int, optional): Generate feature form. 0 means no interactive feature is generated,
                1 means only interactive feature is generated, and 3 means single feature and interactive feature are generated.Defaults to 0.
            include_bias (bool, optional):Whether to include bias column,
                that is, all polynomial powers in this feature are zero.Defaults to False.

        Returns
        -------
            data(pd.DataFrame): The data after deriving.
            model(sklearn.preprocessing.PolynomialFeatures):The model which is used to derive features.

        Usage suggestions
        -----------
            This function is suggested to use for continuous features.
        """
        if feature_interaction not in [0,1,2]:
            raise ValueError("feature_interacation must be 0,1 or 2")
        
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
        Derive data by four arithmetic feature derivation.

        Args
        ----
            data (pd.DataFrame): data need to derived.
            columns (list, optional): features' names that need to derived in the data.
                If columns equals to [], the function will derive all the columns in the data. Defaults to [].
            operations (list, optional): operations, can only contain '+ - * / '. Defaults to ["+","-","*","/"].

        Returns
        -------
            data(pd.DataFrame): the data after deriving.
        """
        for each in operations:
            if each not in ["+","-","*","/"]:
                raise ValueError("operations can only contain '+ - * / ' ")

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
        Derive data by group feature derivation.

        Args
        ----
            data (pd. DataFrame): data that needs to be derived by grouping features.
            maincol (list): The main feature's name, can only have one.
            aggs (list): the parameter of the agg function.
            Parameter aggs format:
                {feature name (str): [feature operation (str),...],...}
                Feature operation support: max,min,mean,var,skew,medium,count,
                nunique,quantile-0.75 (upper quartile),quantile-0.25 (lower quartile),FLF,gold
                ALL-num (operations of all continuous features in common statistics), ALL-cat (operations of all discrete features in common statistics)
            e.g:
            {'Ages':['max', 'mean'], 'Plcass':['min','std']}

        Returns
        -------
        data (pd. DataFrame): all data derived from grouped features, including data without derivating.

        agg parameter interpretation
        ----------------------------
            max, min: maximum value
            mean/var: mean/variance
            skew: deviation of data distribution. It is less than zero and deviates to the left and greater than zero and deviates to the right
            media: median
            quantile: 2/4 quantile
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
        Derive data by target encode derivation.

        Args
        ----
            data (pd.Dataframe): data that need to be derived by target encode derivation(include labels).
            maincol (list): The main feature's name, can only have one.
            label_aggs (dict): The agg function's parameters,
              which can only be processed with labels, are in the same format as group_ Freature_ Derivation() function.
            Parameter label_aggs format:
                {feature name (str): [feature operation (str),...],...}
                Feature operation support: max,min,mean,var,skew,medium,count,
                nunique,quantile-0.75 (upper quartile),quantile-0.25 (lower quartile),FLF,gold
                ALL-num (operations of all continuous features in common statistics), ALL-cat (operations of all discrete features in common statistics)
            e.g:
            {'Ages':['max', 'mean'], 'Plcass':['min','std']}
            
            kfold_length (int, optional): The amount of data for each set of data for which kflod cross-generation features are performed,
             if 0, then no cross-generation features are performed.Defaults to 0.

        Returns
        -------
            data(pd.DataFrame): all data derived , including data without derivating.
        
        Label_agg parameter interpretation
        ----------------------------------
            max, min: maximum value
            mean/var: mean/variance
            skew: deviation of data distribution. It is less than zero and deviates to the left and greater than zero and deviates to the right
            media: median
            quantile: 2/4 quantile
        """
        if kfold_length < 0 or type(kfold_length) == float:
            raise ValueError("kflod_length must be natural number")

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
        Derive data by time feature derivation.

        Args
        ----
            timeSeries (pd.Series): Time series field to be derived from time series characteristics.Defaults time format to "Y-H-D H: M: S".
            time_Stamp (pd.time_stamp, optional): Enter the timestamp of the key time node manually.Defaults to None.
            timeStage parameter format:
                {'Key time node name ': ['Time point']...}, time point default format: "Y-H-D H: M: S".
            high_precision (bool, optional): Whether to perform high-precision (hour, minute, second) calculation.Defaults to False.

        Returns
        -------
            data (pd.DataFrame): data derived from time series characteristics.

        Derivative content of time feature derivation:
        ----------------
            Year, month and day (hour, minute and second) extraction of time point
            The difference between the key time point and the current time point (month, day, (hour, minute, second))
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
    This class is used to following features filter:
        analysis variance(ANOVA): analysis_variance
        feature recursive elimination(RFE): analysis_RFE
    """

    def analysis_variance(self, data:pd.DataFrame, labels:pd.DataFrame)->tuple[pd.DataFrame,feature_selection.SelectKBest]|tuple[pd.DataFrame,feature_selection.SelectKBest,list]:
        """
        Perform variance analysis(ANOVA) on the data, and return the filtered features, filtering model and p_values.
        

        Args
        ----
            data (pd.DataFrame|pd.Series|np.ndarray): data that needs variance analysis.
            labels (pd.DataFrame|pd.Series|np.ndarray): labels of data.

        Returns
        -------
            data (pd.DataFrame): data filtered out after variance analysis, in the order of p_values are from small to large.
            selector (feature_selection.SelectKBest): a trained model.
            p_values (list, optional): p_values of all features.

        Usage suggestions
        -----------------
            variance analysis(ANOVA) is applicable to discrete label filtering for continuous features.

        """
        if data.shape[0] != len(labels):
            raise ValueError("The input data does not match the label")

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
        Recursive feature elimination(RFE) is performed on the data, and the filtered features and filtering model are returned.

        Args
        ----
            data (pd.DataFrame|pd.Series): feature data that requires RFE feature filtering.
            labels (pd.DataFrame|pd.Series): data labels.
            estimator (base.BaseEstimator): Supervised learning estimator used for feature filtering.
            n_jobs (int, optional): The number of CPUs for feature filtering.
                Attention: This parameter is only valid for GridSearchCV.Defaults to 1.
            verbose (int, optional): the detail level of the output model training process during feature filtering.Defaults to 0.
            param (dict, optional): the parameter range of grid search for each training model during feature filtering. Defaults to None (no grid search).
            test_data (pd.DataFrame|None, optional): the feature data used to evaluate the accuracy of the model after each round of feature filter.
                If it is None, the training data will be used for evaluation.Defaults to None
            test_labels (pd.DataFrame|None, optional): data labels used to evaluate the accuracy of the model after each round of feature filter.
                If it is None, the training data will be used for evaluation.Defaults to None.
            high_precision (bool, optional): Whether to perform high-precision search. The default is False.

        Returns
        -------
            data (pd.Dataframe): filtered features.
            scores (list): model score, in the format of [feature name eliminated in this round, model score].
        
        Usage suggestions
        -----------------
            If high_precision is True, it will be more accurate to use RFE features to filter, but it will take up a lot of resources.
            So,it is not recommended to use it on computers with poor configuration.
        """
        if type(test_data) != type(test_labels):
            raise ValueError("test_data and test_labels must be the same type")
        if data.index.shape != labels.index.shape:
            raise ValueError("the number of rows of data must be the same as it of labels")
        if test_data.index.shape != test_labels.index.shape:
            raise ValueError("the number of rows of test_data must be the same as it of test_labels")
        
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

        