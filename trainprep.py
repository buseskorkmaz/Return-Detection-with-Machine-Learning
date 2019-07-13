import pandas as pd
import numpy as np
from datetime import datetime
import random

dataframe = pd.read_csv('traindata.txt', sep=",",header=0)

dataframe = dataframe.drop(dataframe.index[120000:240000])
dataframe = dataframe.reset_index()

# calculate the days between orderDate and deliveryDate
# categorized data if time>7 (1) and if time<7 (0)
deliveryDate = dataframe.loc[:,["deliveryDate"]].values
orderDate = dataframe.loc[:,['orderDate']].values
deliveryTime =np.zeros((dataframe.shape[0],1))
day = []

for i in range(0,dataframe.shape[0]):
    if (deliveryDate[i][0] == '?'):
        continue
    subtractdate = datetime.strptime(deliveryDate[i][0],'%Y-%m-%d') - datetime.strptime(orderDate[i][0],'%Y-%m-%d')
    if(subtractdate.days >0):
        day.append(subtractdate.days)
day = np.asarray(day)
meanDay = int(day.mean())
for i in range(0, dataframe.shape[0]):
    if (deliveryDate[i][0] == '?'):
        if(meanDay < 7):
            deliveryTime[i][0] = 0
        else:
            deliveryTime[i][0] = 1
        continue
    subtractdate = datetime.strptime(deliveryDate[i][0], '%Y-%m-%d') - datetime.strptime(orderDate[i][0], '%Y-%m-%d')
    if (subtractdate.days < 0):
        if(meanDay < 7):
            deliveryTime[i][0] = 0
        else:
            deliveryTime[i][0] = 1
    else:
        if(subtractdate.days < 7):
            deliveryTime[i][0] = 0
        else:
            deliveryTime[i][0] = 1



color = dataframe.loc[:,["color"]].values
colorlist=dataframe['color'].unique()
color_dict = {}
for i,k in enumerate(colorlist):
    color_dict[k] = i

#dataframe['color'].value_counts()
#black 86252 if the color field is empty, we write black because black is the most repeated color.

#Convert color features string to integer values
for i in range(0,dataframe.shape[0]):
    if(color[i][0] == '?'):
        color[i][0] = color_dict['black']
    else:
        color[i][0] = color_dict[color[i][0]]
#    if (color[i][0] == '?'):
#        color[i][0] = np.nan
#    else:
#        color[i][0] = color_dict[color[i][0]]


dateOfBirth = dataframe.loc[:,["dateOfBirth"]].values
age =np.zeros((dataframe.shape[0],1))
age_index = []

for i in range(0,dataframe.shape[0]):
    if(dateOfBirth[i][0] == '?'):
        age_index.append(i)
        #age[i][0] = 0
        continue
    subtractdate = datetime.strptime(orderDate[i][0], '%Y-%m-%d') - datetime.strptime(dateOfBirth[i][0], '%Y-%m-%d')
    if((subtractdate.days / 365)< 35):
        age[i][0] = 0
    elif ((subtractdate.days / 365)< 55):
        age[i][0] = 1
    elif((subtractdate.days / 365)>55):
        age[i][0] = 2

importantDate = np.zeros((dataframe.shape[0],1))
for i in range(0 ,dataframe.shape[0]):
	if(deliveryDate[i][0] == '?' or orderDate[i][0] == '?'):
		continue
	else:
		orderdate = datetime.strptime(orderDate[i][0], '%Y-%m-%d')
		deliverydate = datetime.strptime(deliveryDate[i][0], '%Y-%m-%d')
		if((orderdate.day > 14 and orderdate.month == 1) or (orderdate.month == 2 and orderdate.day < 14)):
			if(deliverydate.day > 14 and (deliverydate.month == 2 or deliverydate.month > 2)):
				importantDate[i][0] = 1
			else:
				importantDate[i][0] = 0
		elif((orderdate.day > 1 and orderdate.month == 12) or (orderdate.month == 12 and orderdate.day < 31)):
			if(deliverydate.day > 1 and deliverydate.month == 1 ):
				importantDate[i][0] = 1
			else:
				importantDate[i][0] = 0
		else:
			importantDate[i][0] = 2


size = dataframe.loc[:,["size"]].values
for i in range(0,dataframe.shape[0]):
	a = size[i][0]
	if((str(a) > '4' and str(a) < '8') or str(a) == 'xs'):
		size[i][0] = 'XS'
	elif((str(a) > '7' and str(a) < '12') or str(a) == 's'):
		size[i][0] = 'S'
	elif((str(a) > '11' and str(a) < '18') or str(a) == 'm'):
		size[i][0] = 'M'
	elif((str(a) > '17' and str(a) < '22') or str(a) == 'l'):
		size[i][0] = 'L'
	elif((str(a) > '21' and str(a) < '26') or str(a) == 'xl'):
		size[i][0] = 'XL'
	elif((str(a) > '25' and str(a) < '30') or str(a) == 'xxl'):
		size[i][0] = 'XXL'
	elif((str(a) > '29' and str(a) < '34') or str(a) == 'xs'):
		size[i][0] = 'XS'
	elif((str(a) > '33' and str(a) < '38') or str(a) == 's'):
		size[i][0] = 'S'
	elif((str(a) > '37' and str(a) < '42') or str(a) == 'm'):
		size[i][0] = 'M'
	elif((str(a) > '41' and str(a) < '46') or str(a) == 'l'):
		size[i][0] = 'L'
	elif((str(a) > '45' and str(a) < '50') or str(a) == 'xl'):
		size[i][0] = 'XL'
	elif((str(a) > '49' and str(a) < '53') or str(a) == 'xxl'):
		size[i][0] = 'XXL'
	elif(str(a) == 'unsized'):
		size[i][0] = np.nan
	else:
		size[i][0] = np.nan

dataframe['size'] = pd.DataFrame.from_records(size, columns = ["size"])

sizelist=dataframe['size'].unique()
size_dict = {}
for i,k in enumerate(sizelist):
    size_dict[k] = i

for i in range(0,dataframe.shape[0]):
    size[i][0] = size_dict[size[i][0]]

salutation = dataframe.loc[:,['salutation']].values
for i in range(0,dataframe.shape[0]):
    if (salutation[i][0] == 'Mrs'):
        salutation[i][0] = 0
    else:
        salutation[i][0] = 1

state = dataframe.loc[:,['state']].values
statelist = dataframe['state'].unique()
state_dict = {}
for i,k in enumerate(statelist):
    state_dict[k] = i
   
for i in range(0,dataframe.shape[0]):
    state[i][0] = state_dict[state[i][0]]
"""
    if(mystate == 'Hamburg'):
        state[i][0] = 0.06
    elif(mystate == 'Bremen'):
        state[i][0] = 0.12
    elif(mystate == 'Bavaria'):
        state[i][0] = 0.18
    elif(mystate == 'Baden-Württemberg'):
        state[i][0] = 0.24
    elif(mystate == 'North Rhine-Westphalia'):
        state[i][0] = 0.3
    elif(mystate == 'Berlin'):
        state[i][0] = 0.36
    elif(mystate == 'Saarland'):
        state[i][0] = 0.42
    elif(mystate == 'Lower Saxony'):
        state[i][0] = 0.48
    elif(mystate == 'Rhineland-Palatinate'):
        state[i][0] = 0.54
    elif(mystate == 'Saxony'):
        state[i][0] = 0.6
    elif(mystate == 'Brandenburg'):
        state[i][0] = 0.66
    elif(mystate == 'Thuringia'):
        state[i][0] = 0.72
    elif(mystate == 'Saxony-Anhalt'):
        state[i][0] = 0.78
    else:
        state[i][0] = 0.84
"""
creationDate = dataframe.loc[:,['creationDate']].values
membershipTime =np.zeros((dataframe.shape[0],1))

for i in range(0,dataframe.shape[0]):
    subtractdate = datetime.strptime(orderDate[i][0],'%Y-%m-%d')- datetime.strptime(creationDate[i][0], '%Y-%m-%d')
    membershipTime[i][0] = subtractdate.days/365

#traindataframe  = pd.read_csv('traindata.txt', sep=",",header=0)
#traindataframe = traindataframe.drop(dataframe.index[120000:240000])
#traindataframe = traindataframe.reset_index()

##itemID new feature extraction
itemID = dataframe.loc[:,['itemID']].values

itemrel = dataframe.groupby(['itemID', 'returnShipment',]).size().unstack(fill_value=0)
item_dict = {}
for index, row in itemrel.iterrows():
    item_dict[index] = row

probab_item ={}
for i in item_dict.keys():
	probab_item[i] = (item_dict[i][1]/(item_dict[i][0] + item_dict[i][1]))

item_prob_one = np.zeros((dataframe.shape[0],1))
for i in range(0,dataframe.shape[0]):
    if(itemID[i][0] in probab_item.keys()):
        item_prob_one[i][0] = probab_item[itemID[i][0]]
    else:
        item_prob_one[i][0] = random.uniform(0,1)

    ##manufacturerID new feature extraction
manufacturerID = dataframe.loc[:,['manufacturerID']].values

manurel = dataframe.groupby(['manufacturerID', 'returnShipment',]).size().unstack(fill_value=0)
manu_dict = {}
for index, row in manurel.iterrows():
    manu_dict[index] = row

probab_manu ={}
for i in manu_dict.keys():
	probab_manu[i] = (manu_dict[i][1]/(manu_dict[i][0] + manu_dict[i][1]))

manu_prob_one = np.zeros((dataframe.shape[0],1))
for i in range(0,dataframe.shape[0]):
    if(manufacturerID[i][0] in probab_manu.keys() ):
        manu_prob_one[i][0] = probab_manu[manufacturerID[i][0]]
    else:
        manu_prob_one[i][0] = random.uniform(0,1)
##customerID new feature extraction
customerID = dataframe.loc[:,['customerID']].values

customerrel = dataframe.groupby(['customerID', 'returnShipment',]).size().unstack(fill_value=0)
customer_dict = {}
for index, row in customerrel.iterrows():
    customer_dict[index] = row

probab_customer ={}
for i in customer_dict.keys():
	probab_customer[i] = (customer_dict[i][1]/(customer_dict[i][0] + customer_dict[i][1]))

customer_prob_one = np.zeros((dataframe.shape[0],1))
for i in range(0,dataframe.shape[0]):
    if(customerID[i][0] in probab_customer.keys() ):
        customer_prob_one[i][0] = probab_customer[customerID[i][0]]
    else:
        customer_prob_one[i][0] = random.uniform(0,1)

size = dataframe.loc[:,['size']].values

for i in range(0,dataframe.shape[0]):
	a = size[i][0]
	if((str(a) > '4' and str(a) < '8') or str(a) == 'xs'):
		size[i][0] = 'XS'
	elif((str(a) > '7' and str(a) < '12') or str(a) == 's'):
		size[i][0] = 'S'
	elif((str(a) > '11' and str(a) < '18') or str(a) == 'm'):
		size[i][0] = 'M'
	elif((str(a) > '17' and str(a) < '22') or str(a) == 'l'):
		size[i][0] = 'L'
	elif((str(a) > '21' and str(a) < '26') or str(a) == 'xl'):
		size[i][0] = 'XL'
	elif((str(a) > '25' and str(a) < '30') or str(a) == 'xxl'):
		size[i][0] = 'XXL'
	elif((str(a) > '29' and str(a) < '34') or str(a) == 'xs'):
		size[i][0] = 'XS'
	elif((str(a) > '33' and str(a) < '38') or str(a) == 's'):
		size[i][0] = 'S'
	elif((str(a) > '37' and str(a) < '42') or str(a) == 'm'):
		size[i][0] = 'M'
	elif((str(a) > '41' and str(a) < '46') or str(a) == 'l'):
		size[i][0] = 'L'
	elif((str(a) > '45' and str(a) < '50') or str(a) == 'xl'):
		size[i][0] = 'XL'
	elif((str(a) > '49' and str(a) < '53') or str(a) == 'xxl'):
		size[i][0] = 'XXL'
	elif(str(a) == 'unsized'):
		size[i][0] = np.nan
	else:
		size[i][0] = np.nan

dataframe['size'] = pd.DataFrame.from_records(size, columns = ["size"])

sizerel = dataframe.groupby(['size','returnShipment',]).size().unstack(fill_value=0)
size_dict = {}
for index,row in sizerel.iterrows():
	size_dict[index] = row

probab_size = {}
for i in size_dict.keys():
	probab_size[i] = (size_dict[i][1]/(size_dict[i][0] + size_dict[i][1]))

size_prob_one = np.zeros((dataframe.shape[0],1))
for i in range(0,dataframe.shape[0]):
    if(size[i][0] in probab_size.keys() ):
        size_prob_one[i][0] = probab_size[size[i][0]]
    else:
        size_prob_one[i][0] = random.uniform(0,1)


newDF = pd.DataFrame()
newDF['orderItemID'] = pd.DataFrame.from_records(dataframe.loc[:,['orderItemID']].values,columns =['orderItemID'])
newDF['deliveryTime'] = pd.DataFrame.from_records(deliveryTime,columns =['deliveryTime'])
newDF['item_prob_one'] = pd.DataFrame.from_records(item_prob_one, columns = ['item_prob_one'])
newDF['size'] = pd.DataFrame.from_records(size_prob_one, columns = ['size'])
newDF['color'] = pd.DataFrame.from_records(color, columns = ['color'])
newDF['manu_prob_one'] = pd.DataFrame.from_records(manu_prob_one, columns = ['manu_prob_one'])
newDF['price'] = pd.DataFrame.from_records(dataframe.loc[:,['price']].values, columns = ['price'])
newDF['customer_prob_one'] = pd.DataFrame.from_records(customer_prob_one, columns = ['customer_prob_one'])
newDF['salutation'] = pd.DataFrame.from_records(salutation, columns = ['salutation'])
newDF['age'] = pd.DataFrame.from_records(age, columns = ['age'])
newDF['state'] = pd.DataFrame.from_records(state, columns = ['state'])
newDF['membershipTime'] = pd.DataFrame.from_records(membershipTime, columns = ['membershipTime'])
newDF['importantDate'] = pd.DataFrame.from_records(importantDate, columns = ['importantDate'])
newDF['returnShipment'] = pd.DataFrame.from_records(dataframe.loc[:,['returnShipment']].values, columns = ['returnShipment'])

newDF = newDF.drop(newDF.index[[age_index]])
newDF = newDF.reset_index(drop=True)
newDF.to_csv("edited_train.csv", sep=',', encoding='utf-8',header=True,index=False)

