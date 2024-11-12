from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("ETTh1.csv", index_col=['date'], parse_dates=['date'],usecols=['date','OT'])
data['moving_avg']=data['OT'].rolling(window=5,min_periods=1).mean()
data['z_score']=(data['OT']-data['OT'].mean())/data['OT'].std()

data.loc[data['z_score'].abs() > 1.5, 'OT'] = data['moving_avg']
print(data.head())

data.plot()
plt.title('original data')
plt.show()