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

"""
Os classificadores usados ao longo do código inteiro
Melhores configurações de hiper-parâmetros and shit \m/
"""
classifiers = {"NN"  : NN(n_neighbors=1),
	           "DMC" : DMC(),
	           "CQG" : CQG()}

"""
Proporcao dos dados para uso no teste
"""
test_rate = 0.3

def reduction_pca(X, y=None, n=None):
	pca = PCA(n_components=n)
	pca.fit(X)
	X = pca.transform(X)
	
	return X

def reduction_lda(X, y, n=None):
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

def classify(classifiers, X, y, test_size, rounds, normalize=False):
	ans = {key: {"score" : [], "sens" : [], "spec" : []}
	       for key, value in classifiers.items()}
	
	for i in range(rounds):
		X_train, X_test, y_train, y_test = data_split(X, y, test_size=test_size)
		for name, classifier in classifiers.items():
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

def find_best_reduction(name, to_run):
	df = pd.read_csv("data/parkinsons.data")
	df = df.drop(["name"], axis=1)
	X = df.drop(["status"], axis=1)
	y = df["status"]
	
	data = {"NN" : [], "DMC" : [], "CQG" : []}
	anses = []
	for n in range(1, len(X.columns)+1):
		X_red = to_run(X, y, n)
		ans = classify(classifiers, X_red, y, test_rate, 100, True)
		anses.append(ans)
		
		data["NN"].append(round(np.mean(ans["NN"]["score"])*100, 2))
		data["DMC"].append(round(np.mean(ans["DMC"]["score"])*100, 2))
		data["CQG"].append(round(np.mean(ans["CQG"]["score"])*100, 2))
		
		#print("{}: shape={}, n={}, OK".format(name, X_red.shape, n))
	
	labels = [str(n) for n in range(1, len(X.columns)+1)]
	df = pd.DataFrame.from_dict(data)
	
	ax = df.plot()
	plt.xticks(np.arange(len(X.columns)+1), labels=labels)
	plt.suptitle("Evolução da precisão em função do {}".format(name))
	plt.xlabel("Número de Componentes")
	plt.ylabel("Precisão (%)")
	plt.ylim((65.0, 90.0))
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)
	
	for d in data:
		max_v = max(data[d])
		n = data[d].index(max_v)
		print("{}: max: {}, n: {}".format(d, max_v, n+1))
		sumary(anses[n])
	
	plt.show()

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
	df = pd.read_csv("data/parkinsons.data")
	df = df.drop(["name"], axis=1)
	X = df.drop(["status"], axis=1)
	y = df["status"]
	
	X = reduction_lda(X, y, 1)
	ans = classify(classifiers, X, y, test_rate, 100, True)
	sumary(ans)
	
	#find_best_reduction("PCA (normalizado)", reduction_pca)
	
	
