import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, pearsonr, spearmanr
import json
import matplotlib.pyplot as plt

#json_filename = "datasets/iclr_2017.json"
json_filename = "datasets/iclr_2017_norm.json"
with open(json_filename) as fd:
        data = json.load(fd)

# load abstract embeddings
npy_filename = "datasets/bert_embeddings.npy"
embeddings = np.load(npy_filename)
norm_idxs = [i for i in map(int,data["id"].keys())]
embeddings = embeddings[norm_idxs,:]

# load acceptances, review scores and citations
acceptances = np.zeros(embeddings.shape[0], dtype=bool)
scores = np.zeros(embeddings.shape[0])
citations = np.zeros(embeddings.shape[0])
for i in range(embeddings.shape[0]):
    acceptances[i] = data["accepted"][str(norm_idxs[i])]
    scores[i] = np.mean(data["recommendation"][str(norm_idxs[i])])
    citations[i] = data["citation"][str(norm_idxs[i])]

save_plot = False

cluster_counts = [1,2,5,10]
#cluster_counts = range(1,11)
cluster_errors = []
print("JSON file: %s" % (json_filename))
print("Embedding file: %s" % (npy_filename))
for num_clusters in cluster_counts:
    print("Running %d clusters" % (num_clusters))

    # kmeans cluster
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=0).fit(embeddings)
    print("K-means inertia: %.2f" % (kmeans.inertia_))
    print()
    cluster_errors.append(kmeans.inertia_)
    
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

    # group metrics
    x,y = [],[]
    acc,rej = [], []
    for i in range(num_clusters):
        acc.append(acceptances[kmeans.labels_==i])
        rej.append(~acceptances[kmeans.labels_==i])
        x.append(scores[kmeans.labels_==i])
        y.append(citations[kmeans.labels_==i])

    # calculate hypotesis testing and correlation
    hyp_tests = np.zeros(num_clusters)
    p_corr = np.zeros(num_clusters)
    s_corr = np.zeros(num_clusters)
    p_pvals = np.zeros(num_clusters)
    s_pvals = np.zeros(num_clusters)
    for i in range(num_clusters):
        _,hyp_tests[i] = ttest_ind(y[i][acc[i]], y[i][rej[i]])
        p_corr[i],p_pvals[i] = pearsonr(x[i],y[i])
        s_corr[i],s_pvals[i] = spearmanr(x[i],y[i])
        
        print("\tHypothesis Testing #%2d  : %.2f               (%d accepted, %d rejected)" % (i, hyp_tests[i], len(y[i][acc[i]]), len(y[i][rej[i]])))
        print("\tPearson Correlation #%2d : %.2f (pval = %.2f) (%d points)" % (i, p_corr[i], p_pvals[i], len(x[i])))
        print("\tSpearman Correlation #%2d: %.2f (pval = %.2f) (%d points)" % (i, s_corr[i], s_pvals[i], len(x[i])))
        print()
    print("\tMean Hypothesis Testing p-value: %.2f" % (hyp_tests.mean()))
    print("\tMean Pearson Correlation       : %.2f (pval = %.2f)" % (p_corr.mean(), p_pvals.mean()))
    print("\tMean Spearman Correlation      : %.2f (pval = %.2f)" % (s_corr.mean(), s_pvals.mean()))
    print()

    if save_plot:
        plt.clf()
        x = np.array(x[0])
        y = np.array(y[0])
        plt.scatter(x[x<100],y[x<100])
        plt.savefig('results/bert_weighted_'+str(num_clusters)+'.png')

"""plt.clf()
plt.plot(cluster_counts,cluster_errors)
plt.savefig('results/kmeans_elbow.png')"""