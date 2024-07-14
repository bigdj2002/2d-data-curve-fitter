import pandas as pd
from scipy import stats

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

file_path = '/mnt/sDisk/tests/test_lm/test_lm/lambda-qp.json'
data = pd.read_json(file_path)

x = data['ax']
y = data['ay']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("\n[scipy]-lineregress 결과:")
print("기울기:", slope)
print("y절편:", intercept)
print("결정계수 R^2:", r_value**2)

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print("\n[statsmodels]-OLS 결과:")
print("기울기:", model.params[1])
print("y절편:", model.params[0])
print("결정계수 R^2:", model.rsquared)

model = LinearRegression().fit(x.values.reshape(-1,1), y)
print("\n[sklearn]-Linear Regression 결과:")
print("기울기:", model.coef_[0])
print("y절편:", model.intercept_)

model = Ridge(alpha=1.0).fit(x.values.reshape(-1,1), y)
print("\n[sklearn]-Ridge Regression 결과:")
print("기울기:", model.coef_[0])
print("y절편:", model.intercept_)

model = Lasso(alpha=1.0).fit(x.values.reshape(-1,1), y)
print("\n[sklearn]-Lasso Regression 결과:")
print("기울기:", model.coef_[0])
print("y절편:", model.intercept_)

model = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(x.values.reshape(-1,1), y)
print("\n[sklearn]-Elastic Net Regression 결과:")
print("기울기:", model.coef_[0])
print("y절편:", model.intercept_)