'''
线性回归————女性身高与体重数据分析
1、数据及分析对象
'women.csv'，数据内容来自The World Almanac and Book of Facts1975。该数据集给出了年龄在30~39岁的15名女性的身高体重数据。(计量单位是inches和pound)

2、目的及分析任务 
l）训练模型
2）对模型进行拟合优度评价和可视化处理，验证简单线性回归建模的有效性
3）采用多项式回归进行模型优化
4）按多项式回归模型预测体重数据

3、方法及工具
用pandas、matplotlib和statsmodels实现
'''
#%%                1.业务理解
'''
本例题所涉及的业务分析女性身高与体重之间的关系，该业务的主要内容是通过建立简单线性回归模型，
然后通过多项式回归进行模型优化，实现依据身高预测一位女性的体重的目的。
'''

#%%                2.数据读取
import pandas as pd
women = pd.read_csv('D:/desktop/ML/回归分析/women.csv',index_col=0)
women

#%%                3.数据理解
#%%% 探索性分析
women.describe()

#%%% 画散点图
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.scatterplot(x='height', y='weight', data=women)
plt.title('height vs weight')
plt.show()
##从散点图可看出，女性身高与体重之间的关系可以视为有线性关系，可做线性回归分析。

#%%                4.数据准备
#先定好自变量X = height和因变量Y = weight
X = women.height
Y = women.weight

#%%                5.模型训练
import statsmodels.api as sm

#%%% 添加常数项
# =============================================================================
# statsmodels需要显式添加常数项来表示截距项（intercept），
# 可以用sm.add_constant()来实现：
# =============================================================================
X = sm.add_constant(X)
print(X)

#%%% 建模，用普通最小二乘法，用statsmodels.OLS()实现
# =============================================================================
# statsmodels.OLS()有4个参数，(endog, exog, missing, hasconst),
# endog因变量y， exog自变量x，
# missing（可选）指定缺失值的处理方式，有none不处理（默认）、drop丢弃和raise提出错误，
# hasconst指定exog中是否包含常数项，默认为False，如果设置为True，则假设已经有一个截距项，并避免再手动添加常数项。
# =============================================================================
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
# =============================================================================
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                 weight   R-squared:                       0.991
# Model:                            OLS   Adj. R-squared:                  0.990
# Method:                 Least Squares   F-statistic:                     1433.
# Date:                Sun, 27 Oct 2024   Prob (F-statistic):           1.09e-14
# Time:                        02:09:16   Log-Likelihood:                -26.541
# No. Observations:                  15   AIC:                             57.08
# Df Residuals:                      13   BIC:                             58.50
# Df Model:                           1                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const        -87.5167      5.937    -14.741      0.000    -100.343     -74.691
# height         3.4500      0.091     37.855      0.000       3.253       3.647
# ==============================================================================
# Omnibus:                        2.396   Durbin-Watson:                   0.315
# Prob(Omnibus):                  0.302   Jarque-Bera (JB):                1.660
# Skew:                           0.789   Prob(JB):                        0.436
# Kurtosis:                       2.596   Cond. No.                         982.
# ==============================================================================

#第二部分中的coef列所对应的const和height就是计算出的回归模型中的截距项b的斜率a

#除了summary()，还可以用params属性来看拟合结果的斜率和截距
print(results.params)
# =============================================================================
# const    -87.516667
# height     3.450000
# =============================================================================

#%%                6.模型评价
#%%% R方
# 以决定系数R^2作为衡量回归直线对观测值拟合程度的指标，取值范围为[0, 1]，
# 越接近1，说明拟合优度越好，调用rsquared属性查看拟合结果的R方：
results.rsquared  #0.9910098

#%%% 可视化
#除了R方等统计量，还可以通过可视化更直观地查看回归效果。
import matplotlib.pyplot as plt
plt.plot(women.height, women.weight,'o')  #观察值
plt.plot(women.height, results.predict(X))  #估计值的回归直线

#%%                7.模型调参tuning parameters
# 从散点图和回归直线可看出，采用简单线性回归的结果效果还可以进一步优化，
# 为此采取多项式回归进行回归分析 （但是在实际项目中，应避免过度优化导致的过拟合）

import numpy as np
X=women.height
Y=women.weight
#%%% 假设体重与身高X、X^2、X^3存在高幂次线性关系，用numpy库的column_stack()创建特征矩阵X_poly
X_poly = np.column_stack((X, np.power(X,2), np.power(X,3)))
print(X_poly)

#%%% 建模拟合
X_poly = sm.add_constant(X_poly)
model_poly = sm.OLS(Y, X_poly)
result_poly = model_poly.fit()
print(result_poly.summary())
# =============================================================================
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                 weight   R-squared:                       1.000
# Model:                            OLS   Adj. R-squared:                  1.000
# Method:                 Least Squares   F-statistic:                 1.679e+04
# Date:                Mon, 28 Oct 2024   Prob (F-statistic):           2.07e-20
# Time:                        18:42:28   Log-Likelihood:                 1.3441
# No. Observations:                  15   AIC:                             5.312
# Df Residuals:                      11   BIC:                             8.144
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const       -896.7476    294.575     -3.044      0.011   -1545.102    -248.393
# x1            46.4108     13.655      3.399      0.006      16.356      76.466
# x2            -0.7462      0.211     -3.544      0.005      -1.210      -0.283
# x3             0.0043      0.001      3.940      0.002       0.002       0.007
# ==============================================================================
# Omnibus:                        0.028   Durbin-Watson:                   2.388
# Prob(Omnibus):                  0.986   Jarque-Bera (JB):                0.127
# Skew:                           0.049   Prob(JB):                        0.939
# Kurtosis:                       2.561   Cond. No.                     1.25e+09
# ==============================================================================

print(result_poly.params)
# =============================================================================
# const   -896.747633
# x1        46.410789
# x2        -0.746184
# x3         0.004253
# =============================================================================
#多项式回归模型中的截距项为-896.7476，而X、X^2、X^3对应的斜率为46.4108、-0.7462和0.0043

#%%% 可视化预测值
plt.plot(women.height, women.weight,'o')  #观察值
plt.plot(women.height, result_poly.predict(X_poly))  #估计值的回归直线
#超级重合，说明采用多项式回归后拟合效果显著提高。











