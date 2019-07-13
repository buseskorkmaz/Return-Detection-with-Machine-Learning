from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('edited_train.csv', sep=",",header=0)
m = df.shape[0]
n = df.shape[1]
X_train = np.zeros((m,n-1))
Y_train = np.zeros((m,1))

#X_train[:,] = df['orderItemID'].values
X_train [:,0] = df['deliveryTime'].values
X_train [:,1] = df['item_prob_one'].values
X_train [:,2] = df['size'].values
#X_train [:,3] = df['color'].values
X_train [:,3] = df['manu_prob_one'].values
X_train [:,4] = df['price'].values
X_train [:,5] = df['customer_prob_one'].values
X_train [:,6] = df['age'].values
X_train [:,7] = df['membershipTime'].values
X_train [:,8] = df['importantDate'].values
Y_train = df['returnShipment'].values


df = pd.read_csv('edited_mytest.csv', sep=",",header=0)
m = df.shape[0]
n = df.shape[1]
X_test = np.zeros((m,n))
Y_test = np.zeros((m,1))

#X_test [:,0] = df['orderItemID'].values
X_test [:,0] = df['deliveryTime'].values
X_test [:,1] = df['item_prob_one'].values
X_test [:,2] = df['size'].values
#X_test [:,3] = df['color'].values
X_test [:,3] = df['manu_prob_one'].values
X_test [:,4] = df['price'].values
X_test [:,5] = df['customer_prob_one'].values
X_test [:,6] = df['age'].values
X_test [:,7] = df['membershipTime'].values
X_test [:,8] = df['importantDate'].values
real = pd.read_csv("mytest.csv",sep=",", header = 0)
Y_test = real['returnShipment'].values

X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)

Y_train = Y_train.astype(np.float)
Y_test = Y_test.astype(np.float)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

def NB_Classifier(X_train, y_train, X_test, y_test):
	gnb_clf = GaussianNB()
	gnb_clf = gnb_clf.fit(X_train, y_train)
	# Evaluate the results with the test set
	predicted = gnb_clf.predict_proba(X_test)
	preds = predicted[:,1]
	fpr = {}
	tpr = {}
	roc_auc = {}
	n_classes = 2
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[i], preds[i])
		roc_auc[i] = auc(fpr[i], tpr[i])
		# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
	return gnb_clf, predicted

def RandomForest_Classifier(X_train, y_train, X_test, k=50000):
    rf_clf = RandomForestClassifier(n_estimators=k)
    rf_clf = rf_clf.fit(X_train, y_train)
    predicted = rf_clf.predict_proba(X_test)

    return rf_clf, predicted

def Logistic(X_train, y_train, X_test):
    log_clf = LogisticRegression()
    log_clf = log_clf.fit(X_train,y_train)
    predicted = log_clf.predict_proba(X_test)

    return log_clf, predicted
    
def SVM(X_train, y_train, X_test):
	svm_clf = svm.SVC()
	svm_clf = svm_clf.fit(X_train,y_train)
	predicted = svm_clf.predict_proba(X_test)
	return svm_clf, predicted
	
def NeuralNetwork(X_train, y_train, X_test):
	neural_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
	neural_clf = neural_clf.fit(X_train, y_train)
	predicted = neural_clf.predict_proba(X_test)
	return neural_clf, predicted
	
def AdaBoost(X_train, y_train, X_test):
	ada_clf = AdaBoostClassifier(n_estimators=200)
	ada_clf = ada_clf.fit(X_train,y_train)
	predicted = ada_clf.predict_proba(X_test)
	return ada_clf, predicted
	
def Bagging(X_train ,y_train, X_test):
	bag_clf = BaggingClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),max_samples=0.5, max_features=0.5)
	bag_clf = bag_clf.fit(X_train,y_train)
	predicted = bag_clf.predict_proba(X_test)
	return bag_clf, predicted

def DecisionTree(X_train, y_train, X_test):
	dec_clf = DecisionTreeClassifier()
	dec_clf = dec_clf.fit(X_train , y_train)
	predicted = dec_clf.predict_proba(X_test)
	return dec_clf, predicted


	

#model,predicted = DecisionTree(X_train_scaled,Y_train,X_test_scaled)
#model,predicted=Bagging(X_train_scaled,Y_train,X_test_scaled)
#model,predicted=RandomForest_Classifier(X_train,Y_train,X_test)
model,predicted=NB_Classifier(X_train_scaled,Y_train,X_test_scaled, Y_test)
#model,predicted=AdaBoost(X_train_scaled,Y_train,X_test_scaled)
#model,predicted=Logistic(X_train_scaled,Y_train,X_test_scaled)
#model,predicted = SVM(X_train_scaled,Y_train,X_test_scaled)
#model,predicted = NeuralNetwork(X_train_scaled,Y_train,X_test_scaled)
result=predicted[:,1]
result= result.reshape(predicted.shape[0],1)
newDF = pd.DataFrame()
newDF['orderItemID'] = pd.DataFrame.from_records(df.loc[:,['orderItemID']].values,columns =['orderItemID'])
newDF['returnShipment'] =pd.DataFrame(result)

newDF.to_csv("sub_deneme.csv", sep=',', encoding='utf-8',header=True,index=False)
