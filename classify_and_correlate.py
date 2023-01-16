import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import json
import matplotlib.pyplot as plt

# load abstract embeddings
npy_filename = "bert_embeddings.npy"
embeddings = np.load(npy_filename)
save_plot = True

cluster_counts = [1,2,5,10]
print("Embedding file: %s" % (npy_filename))
for num_clusters in cluster_counts:
    print("Running %d clusters" % (num_clusters))
    print()
    # kmeans cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

    with open("iclr_2017.json") as fd:
        data = json.load(fd)

    """cluster_titles = [[] for _ in range(5)]
    for i in range(kmeans.labels_.shape[0]):
        cluster_titles[kmeans.labels_[i]].append(data[i]["title"])

    for i in range(5):
        print("Cluster %d:" % (i))
        for s in cluster_titles[i]:
            print("\t%s" % (s))"""

    # plot citations per cluster
    """cluster_citations = [[] for _ in range(5)]
    for i in range(kmeans.labels_.shape[0]):
        if data[i]["citation"] is not None:
            if int(data[i]["citation"])<100:
                cluster_citations[kmeans.labels_[i]].append(int(data[i]["citation"]))

    for i in range(5):
        plt.hist(cluster_citations[i])
        plt.show()"""

    # calculate correlation
    x,y = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    for i in range(kmeans.labels_.shape[0]):
        if data[i]["citation"] is not None:
            x[kmeans.labels_[i]].append(int(data[i]["citation"]))
            #y[kmeans.labels_[i]].append(sum(map(float,data[i]["recommendation"]))/len(data[i]["recommendation"]))
            recommendation = [float(r) for r in data[i]["recommendation"]]
            confidence = [float(c) for c in data[i]["confidence"]]
            y_val = 0
            for j in range(len(recommendation)):
                y_val += recommendation[j] * confidence[j]
            y_val /= sum(confidence)
            y[kmeans.labels_[i]].append(y_val)

    all_corr = np.zeros(num_clusters)
    for i in range(num_clusters):
        all_corr[i] = pearsonr(x[i],y[i])[0]
        print("\tCorrelation #%d: %.2f" % (i, all_corr[i]))
    print("\tMean Correlation: %.2f" % (all_corr.mean()))
    print()

    if save_plot:
        plt.clf()
        x = np.array(x[0])
        y = np.array(y[0])
        plt.scatter(x[x<100],y[x<100])
        plt.savefig('results/bert_weighted_'+str(num_clusters)+'.png')