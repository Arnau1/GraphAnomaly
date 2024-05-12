import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

from collections import Counter


class Cluster_Then_Label():
    def __init__(self, df, classes, n_clusters=8):
        self.data = StandardScaler().fit_transform(df)
        self.y = np.array(classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data,
                                                                                self.y,
                                                                                test_size=0.2)
        self.test_indices = X_test.index.tolist()
        print(self.test_indices)
        self.n_clusters = n_clusters

        print(f"Run KMeans with {self.n_clusters} clusters.")
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', verbose=0).fit(self.data)
               
    
    def get_purities(self):
        print("Find purities")
        purities = [0]*self.n_clusters
        y_pred = np.array(self.kmeans.labels_)
        cluster_labels = np.unique(y_pred)
        # For each cluster label
        for cluster_label in cluster_labels:
            # Get the indices of data points in the cluster
            indices = np.where(y_pred == cluster_label)[0] # [1 3 44 364 2234 3444]
            # Get the ground truth labels of these data points
            # Discard labels in the train set as unknown
            ground_truth_train = np.array([-1 for i, label in enumerate(self.y) if i in self.test_indices])
            cluster_labels_true = ground_truth_train[indices] # [-1 -1 1 -1 1 1]
            # Find the majority and minority class in the cluster
            counts = Counter(cluster_labels_true)#.most_common(1)[0][1]
            # Save the max class as well as the counts
            max_count = max(counts[0], counts[1])
            max_class = list(counts.keys())[list(counts.values()).index(max_count)]
            min_count = min(counts[0], counts[1],)
            # Save the majority labelled class and its purity for each cluster
            purities[cluster_label] = (max_class, max_count/(max_count+min_count))
        return purities                   
            
    
    def pseudo_label(self, n_iters=5):
        ...

    def test(self):
        ...
