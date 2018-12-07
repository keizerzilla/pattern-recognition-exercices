"""
tc3.py
Artur Rodrigues Rocha Neto
artur.rodrigues26@gmail.com
DEZ/2018

==========================
Trabalho Computacional #03
==========================

Sobre: clusterização de dados desconhecidos
Requisitos: recpad.py
"""

from recpad import *

sb.set()

dataset = "data/datasetTC3.dat"
cols = ["atrib{}".format(n) for n in range(1, 7)]

df = pd.read_csv(dataset, header=None)
df.columns = cols
X = np.array(df)
X_trans = super_normalize(X)
X_trans = super_unskew(X_trans)
X_trans = pd.DataFrame(X_trans, columns=cols)
"""
#================
# Sobre os dados
#================

heatmap(df, "Correlação nos dados desconhecidos")
stats = data_stats(df)
stats.to_csv("data/tc3-datastats.csv")
pairplot(df, "Relacionamentos par-a-par entre os atributos")
"""
#=====================================================================
# Calculando indices de validacao para diferentes numeros de clusters
#=====================================================================
"""
data = {"n" : [], "ch" : [], "db" : [], "dunn" : []}
for n in range(2, 21):
	ch, db, dunn, _, _ = clustering_kmeans(X, n)
	
	data["n"].append(n)
	data["ch"].append(ch)
	data["dunn"].append(dunn)
	data["db"].append(db)
	
	print("n={}, ch={}, db={}, dunn={}".format(n, ch, db, dunn))

dump = pd.DataFrame.from_dict(data)

dump.plot(x="n", y=["dunn", "db"])
plt.title("Índices Dunn e Davies-Bouldin para diferentes K")
plt.xlabel("K")
plt.ylabel("Valor")
plt.xticks([n for n in range(2, 21)])
plt.show()

dump.plot(x="n", y=["ch"])
plt.title("Índice Calinski-Harabaz para diferentes K")
plt.xlabel("K")
plt.ylabel("Valor")
plt.xticks([n for n in range(2, 21)])
plt.show()

#===========================================================
# Calculando indices de validacao (conjunto pre-processado)
#===========================================================

data = {"n" : [], "ch" : [], "db" : [], "dunn" : []}
for n in range(2, 21):
	ch, db, dunn, _, _ = clustering_kmeans(X_trans, n)
	
	data["n"].append(n)
	data["ch"].append(ch)
	data["dunn"].append(dunn)
	data["db"].append(db)
	
	print("n={}, ch={}, db={}, dunn={}".format(n, ch, db, dunn))

dump = pd.DataFrame.from_dict(data)

dump.plot(x="n", y=["ch"])
plt.title("Índice Calinski-Harabaz para diferentes K")
plt.xlabel("K")
plt.ylabel("Valor")
plt.xticks([n for n in range(2, 21)])
plt.show()
"""

#=====================
# analise estatistica
#=====================

ch, db, dunn, labels, centroids = clustering_kmeans(X, 2)
centroids = pd.DataFrame(centroids, columns=cols)
stats = data_stats(centroids)
stats.to_csv("data/tc3-analise-estatistica.csv")

uni, count = np.unique(labels, return_counts=True)
print("{} {}".format(uni, count))


























