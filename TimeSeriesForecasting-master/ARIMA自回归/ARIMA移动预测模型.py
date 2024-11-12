from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
X = series.values
size = int(len(X) * 0.66)
train_data =series.iloc[:size].copy()
test_data = series.iloc[size:].copy()
train_data= scaler_train.fit_transform(train_data)
test_data = scaler_test.fit_transform(test_data)
# train, test = X[0:size], X[size:len(X)]
history=[x for x in train_data]
test =[x for x in test_data]
print(history)
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_absolute_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()