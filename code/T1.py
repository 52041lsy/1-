import pandas as pd
import pulp as pp
import numpy as np

#参数引入
#国产，对应2
Q_dm=pd.read_csv('data_p/Q_dm.csv')
Q_dm=Q_dm.to_numpy()
Q_dm=Q_dm[:,1]
Q_dm=Q_dm.reshape((1,106))
Q_dm=Q_dm.tolist()
print(Q_dm)