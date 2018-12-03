"""
recpad.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
NOV/2018

Sobre: conjunto de funções criadas para a resolução dos trabalhos computacionais
Requisitos: Python 3.5+, numpy, pandas, matplotlib, scikit-learn, seaborn
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid as DMC
from sklearn.neighbors import KNeighborsClassifier as NN
from sklearn.model_selection import train_test_split as data_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as CQG

def kmeans_cluster(X, n=1000):
	X_new, _, _ = k_means(X, n_clusters=n, n_jobs=-1)
	
	return X_new

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
			confmatrix = confusion_matrix(y_test, y_pred)
			tn, fp, fn, tp = confmatrix.ravel()
			sens = sensitivity(tp, fn)
			spec = specificity(tn, fp)
			
			ans[name]["score"].append(score)
			ans[name]["sens"].append(sens)
			ans[name]["spec"].append(spec)
			
			#sb.heatmap(confmatrix, cmap="Purples", annot=True, fmt="d")
			#plt.title("classificador: {}".format(name))
			#plt.show()
	
	return ans

def find_best_pca(dataset, classifiers, test_rate, save_plot):
	df = pd.read_csv(dataset)
	df = df.drop(["name"], axis=1)
	X = df.drop(["status"], axis=1)
	y = df["status"]
	
	data = {"NN" : [], "DMC" : [], "CQG" : []}
	anses = []
	for n in range(1, len(X.columns)+1):
		X_red = reduction_pca(X, y, n)
		ans = classify(classifiers, X_red, y, test_rate, 100, True)
		anses.append(ans)
		
		data["NN"].append(round(np.mean(ans["NN"]["score"])*100, 2))
		data["DMC"].append(round(np.mean(ans["DMC"]["score"])*100, 2))
		data["CQG"].append(round(np.mean(ans["CQG"]["score"])*100, 2))
	
	labels = [str(n) for n in range(1, len(X.columns)+1)]
	df = pd.DataFrame.from_dict(data)
	
	ax = df.plot()
	plt.xticks(np.arange(len(X.columns)+1), labels=labels)
	plt.suptitle("Evolução da precisão em função do PCA")
	plt.xlabel("Número de Componentes")
	plt.ylabel("Precisão (%)")
	plt.ylim((65.0, 95.0))
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)
	
	for d in data:
		max_v = max(data[d])
		n = data[d].index(max_v)
		new_ans = {d : anses[n][d]}
		sumary(new_ans, "PCA: classif {}, q={}, max={}".format(d, n+1, max_v))
	
	plt.savefig(save_plot)

def tc2_reduction_EDITAR():
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

def sumary(ans, title):
	size = 70
	separator = "-"
	
	print(separator*size)
	print("SUMARY: {}".format(title))
	print(separator*size)
	print("CLASSIF\t\tMEAN\tMEDIAN\tMINV\tMAXV\tSTD\tSENS\tSPEC")
	print(separator*size)
	
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
	
	print(separator*size)
	print()
	
