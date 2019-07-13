import pandas as pd
import math
import numpy as np

estimate = pd.read_csv("sub_deneme.csv", sep=",",header=0)

real = pd.read_csv("mytest.csv",sep=",", header = 0)

def error(real,estimate):
	return np.abs(real-estimate)

realreturn = real.loc[:,['returnShipment']].values
estimatereturn = estimate.loc[:,['returnShipment']].values

test_error = np.zeros((estimate.shape[0],1))
for i in range(0,estimate.shape[0]):
	test_error[i][0] = error(realreturn[i][0], estimatereturn[i][0])

sum = 0
for i in range(0, estimate.shape[0]):
	sum = sum + test_error[i][0]

print("Error: " + str(sum))
