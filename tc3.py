from recpad import *

dataset = "data/datasetTC3.dat"
cols = ["atrib{}".format(n) for n in range(1, 7)]

df = pd.read_csv(dataset, header=None)
df.columns = cols
X = np.array(df)

#=============================================
# Relacionamento par-a-par entre os atributos
#=============================================

pairplot(df, "Relacionamentos par-a-par entre os atributos")

#=============================================================
# Relacionamento par-a-par entre os atributos (transformados)
#=============================================================

X_trans = super_normalize(X)
X_trans = super_unskew(X_trans)
trans = pd.DataFrame(X_trans, columns=cols)

sb.pairplot(trans, markers="o", palette="GnBu_d", diag_kind="kde",
            plot_kws=dict(s=50, edgecolor="b", linewidth=1))
plt.subplots_adjust(left=0.06, bottom=0.07, right=0.98, top=0.92, wspace=0.18,
                    hspace=0.18)
plt.suptitle("Relacionamentos par-a-par entre os atributos (transformados)")
plt.savefig("figures/tc3/tc3-pairplot-transformados.png")
plt.close()

#=====================
# Estatisticas gerais
#=====================

stats = data_stats(df)
stats.to_csv("data/tc3-datastats.csv", index=False)

#=====================================================================
# Calculando indices de validacao para diferentes numeros de clusters
#=====================================================================

data = {"n" : [], "ch" : [], "db" : [], "dunn" : []}
for n in range(2, 40):
	ch, db, dunn = clustering_kmeans(X, n)
	
	data["n"].append(n)
	data["ch"].append(ch)
	data["dunn"].append(dunn)
	data["db"].append(db)
	
	print("n={}, ch={}, db={}, dunn={}".format(n, ch, db, dunn))

dump = pd.DataFrame.from_dict(data)
dump.to_csv("data/tc3-ch-dunn.txt", index=False)

dump.plot(x="n", y=["db", "ch", "dunn"])
plt.show()

dump.plot(x="n", y=["db", "dunn"])
plt.show()

