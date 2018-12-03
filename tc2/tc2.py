"""
tcc2.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
NOV/2018

Código-fonte do Trabalho Computacional #02 de Reconhecimento de Padrões 2018.2
Requisitos: Python 3.5+, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB as CQG
from sklearn.neighbors import NearestCentroid as DMC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as data_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def kmeans_cluster(X, n=1000):
	X_new, _, _ = k_means(X, n_clusters=n, n_jobs=-1)
	
	return X_new

def sensitivity(tp, fn):
	return tp / (tp + fn)

def specificity(tn, fp):
	return tn / (tn + fp)

def do_normalize(X_train, X_test):
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	
	return X_train, X_test

def classify(classifiers, X, y, test_size, rounds, normalize=False):
	ans = {key: {"score" : [], "sens" : [], "spec" : []}
	       for key, value in classifiers.items()}
	
	for i in range(rounds):
		for name, classifier in classifiers.items():
			X_train, X_test, y_train, y_test = data_split(X, y,
			                                              test_size=test_size,
			                                              shuffle=True)
			
			if normalize:
				X_train, X_test = do_normalize(X_train, X_test)
			
			classifier.fit(X_train, y_train)
			score = classifier.score(X_test, y_test)
			y_pred = classifier.predict(X_test)
			tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
			sens = sensitivity(tp, fn)
			spec = specificity(tn, fp)
			
			ans[name]["score"].append(score)
			ans[name]["sens"].append(sens)
			ans[name]["spec"].append(spec)
	
	return ans

def tcc2_reduction():
	df = pd.read_csv("data/default-of-credit-card-clients.csv")
	df = df.drop(["ID"], axis=1)
	X0 = df.loc[df["default-payment-next-month"] == 0]
	X0 = X0.drop(["default-payment-next-month"], axis=1)
	X1 = df.loc[df["default-payment-next-month"] == 1]
	X1 = X1.drop(["default-payment-next-month"], axis=1)
	
	print("shape class [0] pre-kmeans: {}".format(X0.shape))
	print("shape class [1] pre-kmeans: {}".format(X1.shape))
	
	print("clustering X0"); X0_new = kmeans_cluster(X0); print("X0 done");
	print("clustering X1"); X1_new = kmeans_cluster(X1); print("X1 done");
	
	print("shape class [0] pos-kmeans: {}".format(X0_new.shape))
	print("shape class [1] pos-kmeans: {}".format(X1_new.shape))
	
	X0 = pd.DataFrame(data=X0_new)
	X1 = pd.DataFrame(data=X1_new)
	y0 = pd.DataFrame(data=np.zeros((len(X0.index), ), dtype=np.int))
	y1 = pd.DataFrame(data=np.ones((len(X1.index), ), dtype=np.int))
	X = pd.concat([X0, X1])
	y = pd.concat([y0, y1])
	
	X0.to_csv("data/X0.csv", index=False)
	X1.to_csv("data/X1.csv", index=False)
	X.to_csv("data/X.csv", index=False)
	y.to_csv("data/y.csv", index=False)
	print("clustered data saved")

def sumary(ans, title="SUMARY"):
	size = 70
	print("-"*size)
	print("[[ {} ]]".format(title))
	print("-"*size)
	print("CLASSIF\t\tMEAN\tMEDIAN\tMINV\tMAXV\tSTD\tSENS\tSPEC")
	print("-"*size)
	
	for n in ans:
		mean = round(np.mean(ans[n]["score"])*100, 2)
		median = round(np.median(ans[n]["score"])*100, 2)
		minv = round(np.min(ans[n]["score"])*100, 2)
		maxv = round(np.max(ans[n]["score"])*100, 2)
		std = round(np.std(ans[n]["score"])*100, 2)
		sens = round(np.mean(ans[n]["sens"])*100, 2)
		spec = round(np.mean(ans[n]["spec"])*100, 2)
		print("{}\t\t{}\t{}\t{}\t{}\t{}\t{}%\t{}%".format(n, mean, median, minv,
		                                                  maxv, std, sens,
		                                                  spec))
	
	print("-"*size)
	print()

if __name__ == "__main__":
	classifiers = {"KNN" : KNN(n_neighbors=1),
	               "DMC" : DMC(),
	               "CQG" : CQG()}
	
	df = pd.read_csv("data/default-of-credit-card-clients.csv")
	df = df.drop(["ID"], axis=1)
	X = df.drop(["default-payment-next-month"], axis=1)
	y = df["default-payment-next-month"]
	ans = classify(classifiers, X, y, 0.3, 100)
	sumary(ans, "ORIGIAL - SEM NORMALIZAR")
	
	X0 = pd.read_csv("data/X0.csv")
	X1 = pd.read_csv("data/X1.csv")
	X = pd.concat([X0, X1])
	y = np.ravel(pd.read_csv("data/y.csv"))
	ans = classify(classifiers, X, y, 0.3, 100)
	sumary(ans, "REDUZIDO - SEM NORMALIZAR")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
