"""
tcc1.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
NOV/2018

Código-fonte do Trabalho Computacional #01 de Reconhecimento de Padrões 2018.2
Requisitos: Python 3.5+, numpy, pandas, matplotlib, scikit-learn, seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB as CQG
from sklearn.neighbors import NearestCentroid as DMC
from sklearn.neighbors import KNeighborsClassifier as NN
from sklearn.model_selection import train_test_split as data_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def reduction_pca(X, n):
	pca = PCA(n_components=n)
	pca.fit(X)
	X = pca.transform(X)
	
	return X

def reduction_lda(X, y, n):
	pca = LDA(n_components=n)
	pca.fit(X, y)
	X = pca.transform(X)
	
	return X

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

def classify(classifiers, X, y, test_size, rounds, normalize):
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
	
def sumary(ans):
	size = 70
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

if __name__ == "__main__":
	classifiers = {"NN"  : NN(n_neighbors=1),
	               "DMC" : DMC(),
	               "CQG" : CQG()}
	
	df = pd.read_csv("data/parkinsons.data")
	df = df.drop(["name"], axis=1)
	X = df.drop(["status"], axis=1)
	y = df["status"]
	
	print("SEM NORMALIZACAO")
	ans = classify(classifiers, X, y, 0.3, 100, False)
	sumary(ans)
	
	print("COM NORMALIZACAO")
	ans = classify(classifiers, X, y, 0.3, 100, True)
	sumary(ans)
	
