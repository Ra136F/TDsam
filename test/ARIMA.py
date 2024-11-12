import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))

data=pd.read_csv("ETTh1.csv", index_col=['date'], parse_dates=['date'],usecols=['date','OT'])
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size].copy()
test_data = data.iloc[train_size:].copy()

train_data['OT']= scaler_train.fit_transform(train_data)
test_data['OT'] = scaler_test.fit_transform(test_data)
# from statsmodels.tsa.stattools import adfuller
#
# # 对训练集进行ADF检验
# adf_result = adfuller(train_data['OT'])
#
# # 输出结果
# print("ADF 检验统计量:", adf_result[0])
# print("p 值:", adf_result[1])
# print("临界值:", adf_result[4])
#
# # 判断是否平稳
# if adf_result[1] < 0.05:
#     print("训练集时间序列是平稳的（在5%的显著性水平上）")
# else:
#     print("训练集时间序列非平稳，可以考虑进行差分转换")


# pvalue=acorr_ljungbox(train_data['OT'], lags=1)
# print(pvalue)

# model = pm.auto_arima(train_data)
# y_pred=model.predict(int(len(data) * 0.2))
# model = ARIMA(train_data['OT'], order=(4, 1, 1))
# model_fit=model.fit()
# predictions = model_fit.predict(0, end=len(test_data)-1)
# print(predictions.head())
# mae = mean_absolute_error(test_data['OT'], predictions)
# print("mean_absolute_error： ", mae)

history = [x for x in train_data['OT']]
predictions = list()

y=[x for x in test_data['OT']]

for t in range(len(test_data)):
	model = ARIMA(history, order=(4,1,1))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = y[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_absolute_error(test_data, predictions)
print("mean_absolute_error： ",error)