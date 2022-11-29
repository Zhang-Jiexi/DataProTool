DataProTool
============

introduction
-------------

It is a library that support advance tools in feature engineering and data progress. </br>
This library is independently developed by Zhang Jiexi. Author's e-mail: <zhangjiexi66696@outlook.com>

install
--------

you can use `pip install DataProTool` to install this library. </br>
in version `1.2.0` , I rewrite the setup file so you don't have to install dependent libraries manually.

what's new in the version`1.2.0`
---------------------------------

1.Support English!</br>
Now, I create an English version for this tool.and I set this tool's main language is English.</br>
You can simplely use `import dataprotool`.</br>
If you want to use Chinese version, you can use `import dataprotool.cn` to import.

2.I add three feature derivation function in the `FreatureDerivation` class, they are:</br>
        1.target encode derivation: `target_encode_derivation()`</br>
        2.four arithmetic feature derivation: `four_arithmetic_feature_derivation()`</br>
        3.cross combination feature derivation: `cross_combination_feature_derivation()`</br>

3.When you use the function of `FeatureFilter` class ,it will return the score or sort of data.
4.Fix some bugs in some function.

dependent libraries
------------------------

numpy ~= 1.23.3
pandas ~= 1.5.0
scikit-learn ~= 1.0.2
tqdm ~= 4.64.1

`tqdm` is used to create progress bar.
