import pandas as pd

dataframe = pd.read_csv('traindata.txt', sep=",",header=0)

newdataframe = dataframe.loc[120000:240000,:]
newdataframe.to_csv("mytest.csv", sep=',', encoding='utf-8',header=True,index=False)
