import numpy as np # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.cluster import KMeans, DBSCAN # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import * #type: ignore

from collections import Counter
import random


class Cluster_Then_Label():
    def __init__(self, df, classes, n_clusters=8, method='kmeans', random_state=None):
        assert method in ['kmeans', 'dbscan'], "Error: Unknown method"
        # Standarize data (np array)
        self.data = StandardScaler().fit_transform(df)
        # Class labels
        self.y = np.array(classes)
        # Indices
        self.df_indices = df.index.tolist()
        self.indices = np.arange(0, len(self.y))
        # Train test split saving the indices
        if random_state is None:
            random_state = random.randint(0,100000)
        self.random_state = random_state
        self.train_indices, self.test_indices = \
            train_test_split(self.indices, test_size=0.2, random_state=self.random_state)
        # Unlabelling the test datapoints
        self.ground_truth_train = np.array([-1 if self.indices[i] in self.test_indices else label for i, label in enumerate(self.y)])
        # Clustering method
        self.method = method
        # Number of clusters
        self.k = n_clusters
        # Other array initializations
        self.y_pred = np.empty(0)
        self.purities = [0]*self.k # Array [0,0,0...]
        self.idx = {}
        self.clustering = None

    def run_clustering(self):
        if self.method == 'kmeans':
            print(f"Run KMeans with {self.k} clusters.")
            self.clustering = KMeans(n_clusters=self.k, n_init='auto', verbose=0).fit(self.data)
        elif self.method == 'dbscan':
            self.clustering = DBSCAN(eps=13).fit(self.data)
            if -1 in list(self.clustering.labels_):
                self.k = len(np.unique(self.clustering.labels_)) - 1
            else:
                self.k = len(np.unique(self.clustering.labels_))
           
    
    def get_purities(self):
        # print("Find purities")
        purities = [0]*self.k
        # Predicted cluster labels
        self.y_pred = np.array(self.clustering.labels_)
        cluster_labels = list(np.unique(self.y_pred))
        # Manage points of no class
        if -1 in cluster_labels:
            cluster_labels.remove(-1)
        # For each cluster label
        for cluster_label in cluster_labels:
            # Get the indices of data points in the cluster
            self.idx[cluster_label] = np.where(self.y_pred == cluster_label)[0]
            # Discard labels in the train set as unknown
            cluster_labels_true = self.ground_truth_train[self.idx[cluster_label]] # [-1 -1 1 -1 1 1]
            # Find the majority and minority class in the cluster
            counts = Counter(cluster_labels_true) #.most_common(1)[0][1]
            # Add 0 values to avoid errors
            if list(counts.keys()) == [-1]:
                counts[0] = 0
                counts[1] = 0
            clean_counts = counts.copy()
            # Drop -1 values because we don't care about them
            if -1 in list(clean_counts.keys()):
                del clean_counts[-1]
            # Save the max class as well as the counts
            max_count = max(counts[0], counts[1]) # maximum between class 0 and class 1
            max_class = list(counts.keys())[list(counts.values()).index(max_count)]
            min_count = min(counts[0], counts[1])
            # Save the majority labelled class and its purity for each cluster
            # Add a small number to avoid division by zero error
            purities[cluster_label] = (max_class, max_count/(max_count+min_count+1e-10))
        self.purities = purities 
    
    def loop(self, n_iters=1):
        # Create counter of conversions
        conversions = [0, 0]
        for iter in range(n_iters):
            # Array to save from which cluster we want to substitute unknowns and to which value
            # [(cluster_label, class_label)]
            substitute = []
            # Run k means and obtain the purities for each cluster
            self.run_clustering()
            self.get_purities()
            # For each cluster, if 85% of labeled nodes are of the same class, save the label and the class
            for c_label in range(len(self.purities)):
                if self.purities[c_label][1] > 0.85:
                    substitute.append((c_label, self.purities[c_label][0]))
            # If no cluster filled the condition, end this iteration
            if len(substitute) == 0:
                break
            # For each cluster and label, substitute from ground_truth_train the unknown nodes by their 
            # new labels. Also, count when this happens for label 0 and when for label 1.
            for cluster_index, label in substitute:
                slice = self.ground_truth_train[self.idx[cluster_index]]
                slice[slice == -1] = label
                self.ground_truth_train[self.idx[cluster_index]] = slice
                conversions[label] += 1
        # print(substitute)
        # print(self.purities, iter)
        print(f"After {iter+1} iterations:\n- {conversions[0]} nodes were pseudo-labelled as 0.\n- {conversions[1]} node(s) were pseudo labelled as 1.\n")

    def test(self, display_conf_matrix=False):
        # Obtain test slices
        y_val = self.y[self.test_indices]
        y_pred = self.ground_truth_train[self.test_indices]
        # Get rid of unknown values
        y_val = np.array([element for element in y_val])
        y_pred = np.array([element for element in y_pred])
        # Generate and show confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=[-1, 0, 1])
        if display_conf_matrix:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Unknown', 'Licit', 'Ilicit'])
            disp.plot()
            plt.title("Cluster-then-label")
            plt.show()
        # Recall = TP / (TP + FN)
        recall = cm[2,2] / (cm[2,2] + cm[2,1] + cm[2,0] + 1e-10)
        # F1 score = 2TP / (2TP + FN + FP)
        f1 = 2*cm[2,2] / (2*cm[2,2] + cm[1,2] + cm[2,1] + cm[2,0] + 1e-10)
        print(f"Recall: {recall}\nF1 score: {f1}")