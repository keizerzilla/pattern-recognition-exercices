"""
tc2.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
DEZ/2018

==========================
Trabalho Computacional #02
==========================

Sobre: quantizacao vetorial para reducao de grandes volumes de dados em
problemas de classificacao
Requisitos: recpad.py
"""

from recpad import *

classifiers = {"NN"  : NN(n_neighbors=1),
               "DMC" : DMC(),
               "CQG" : CQG(store_covariance=True)}
dataset = "data/default-of-credit-card-clients.csv"
test_size = 0.3
rounds = 100
default_n_clusters = 1000

df = pd.read_csv(dataset)
df = df.drop(["ID"], axis=1)
X = df.drop(["default-payment-next-month"], axis=1)
y = df["default-payment-next-month"]

X_c0 = df.loc[df["default-payment-next-month"] == 0]
X_c0 = X_c0.drop(["default-payment-next-month"], axis=1)
X_c1 = df.loc[df["default-payment-next-month"] == 1]
X_c1 = X_c1.drop(["default-payment-next-month"], axis=1)

print("Amostras da classe 0: {}".format(X_c0.shape))
print("Amostras da classe 1: {}".format(X_c1.shape))

ans = classify(classifiers, X, y, test_size, rounds, verbose=True)
sumary(ans, "TC2 - classificacao sem reducao de dados")

data = {"NN" : [], "DMC" : [], "CQG" : []}
n_init = 200
n_end = 6200
n_step = 200
ticks = int((n_end - n_init) / n_step)
vector = range(n_init, n_end, n_step)

print("Numero clusters testados: {}".format(ticks))
for n in vector:
	print("EXECUTANDO KMEDIAS (n_clusters = {})".format(n))

	print("> classe 0...")
	X_c0_red = kmeans_cluster(X_c0, n=n)
	print("kmedias OK! Shape de X_red_c0: {}".format(X_c0_red.shape))

	print("> classe 1...")
	X_c1_red = kmeans_cluster(X_c1, n=n)
	print("kmedias OK! Shape de X_red_c1: {}".format(X_c1_red.shape))

	X_red = np.concatenate((X_c0_red, X_c1_red))
	print("Shape de X_red: {}".format(X_red.shape))

	y_c0 = np.zeros((n, ), dtype=int)
	y_c1 = np.ones((n, ), dtype=int)
	y_red = np.concatenate((y_c0, y_c1))
	print("Shape de y_c0: {}".format(y_c0.shape))
	print("Shape de y_c1: {}".format(y_c1.shape))
	print("Shape de y_red: {}".format(y_red.shape))

	ans = classify(classifiers, X_red, y_red, test_size, rounds, verbose=True)
	
	nn = round(np.mean(ans["NN"]["score"])*100, 2)
	dmc = round(np.mean(ans["DMC"]["score"])*100, 2)
	cqg = round(np.mean(ans["CQG"]["score"])*100, 2)
	
	data["NN"].append(nn)
	data["DMC"].append(dmc)
	data["CQG"].append(cqg)
	
	print()
	print("[n = {}]: nn = {}, dmc = {}, cqg = {}".format(n, nn, dmc, cqg))
	
labels = [str(n) for n in vector]
df = pd.DataFrame.from_dict(data)

ax = df.plot()
plt.xticks(np.arange(ticks), labels=labels)
plt.suptitle("Evolução da precisão em função do número de clusters")
plt.xlabel("Número de clusters")
plt.ylabel("Precisão (%)")
plt.ylim((0.0, 100.0))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3)
plt.savefig("figures/tc2-evolucao-clusters.png")
plt.show()

