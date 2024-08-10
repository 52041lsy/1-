import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('C:/Users/86183/Desktop/input_data/df_past_order2.csv')
forecast_range = 7
forecasts = {}
rmses = {}

# 对每一行数据分别进行预测
for index, row in df.iterrows():
    data = row.values
    train_data = data[:-forecast_range]
    
    # 定义 ARIMA 模型 (p, d, q) 参数
    p = 5  # AR 级数
    d = 1  # 差分次数
    q = 0  # MA 级数
    
    # 训练 ARIMA 模型
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()
    
    # 预测未来7天的值
    forecast = model_fit.forecast(steps=forecast_range)
    forecasts[row[0]] = forecast  # 使用城市名作为键存储预测结果
    
    # 计算 RMSE
    actual = data[-forecast_range:]  # 实际值
    rmse = sqrt(mean_squared_error(actual, forecast))
    rmses[row[0]] = rmse

# 输出每个城市的 RMSE 和预测结果
for city in forecasts.keys():
    print(f'City: {city}')
    print(f'RMSE: {rmses[city]}')
    print(f'Forecast for 6.24-6.30: {forecasts[city]}')
