from recpad import *

dataset = "data/datasetTC3.dat"

df = pd.read_csv(dataset, header=None)
X = np.array(df)

X = super_normalize(X)

data = {"n" : [], "ch" : [], "dunn" : []}
for n in range(2, 20):
	ch, dunn = clustering_kmeans(X, n)
	
	data["n"].append(n)
	data["ch"].append(ch)
	data["dunn"].append(dunn)
	
	print("n={}, ch={}, dunn={}".format(n, ch, dunn))

dump = pd.DataFrame.from_dict(data)
dump.to_csv("data/tc3_dump.txt", index=False)

dump.plot(x="n")
plt.xticks([str(n) for n in range(2, 20)])
plt.show()
