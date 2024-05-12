import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# The class from the ML_algorithms notebook
class machineLearning:
    def __init__(self, df):
        self.df = df
        self.results = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1 score"])
        self.df_pca = None
    def apply_PCA(self, ncomponents):
        # Standardize the features (important for PCA)
        scaler = StandardScaler()
        
        if 'class' in self.df.columns:
            df_pca = self.df.drop(columns=['class'])
        else:
            df_pca = self.df
        scaled_data = scaler.fit_transform(df_pca)

        # Apply PCA
        pca = PCA(n_components=ncomponents)  # You can choose the number of components you want to keep
        principal_components = pca.fit_transform(scaled_data)

        # Create a DataFrame for the principal components
        columns = [f"PC{i+1}" for i in range(principal_components.shape[1])]
        principal_df = pd.DataFrame(data=principal_components, columns=columns)


        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = explained_variance_ratio.sum()

        print(f"\nExplained variance ratio: {cumulative_variance_ratio}")
        print(f"Data reduction, from shape {df_pca.shape} to {principal_df.shape}")
        
        # Add two columns to be able to apply ML models later on
        principal_df['txId'] = df_pca.index
        principal_df['class'] = list(self.df['class'])
        
        self.df_pca = principal_df
        return principal_df
    
    
    def train_and_test(self, algorithm, use_pca=True, display_conf_matrix=False, modeMarti=False):
        # Using node2vec we are don't need a PCA because we aro not working with that many dimensions
        if use_pca:
            df = self.df_pca
        else:
            df = self.df
        # Select the type of class labeling
        if modeMarti:
            X = df.loc[df['class'].isin([0,1])].drop(columns=['txId', 'class'])
            y = df.loc[df['class'].isin([0,1])]['class']
        else:
            X = df.loc[df['class'].isin(['1', '2'])].drop(columns=['txId', 'class'])
            y = df.loc[df['class'].isin(['1', '2'])]['class']
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Select model
        print(f"\nTraining {algorithm}...\n")
        try:
            if algorithm == "Logistic regression":
                model = LogisticRegression()
                
            elif algorithm == "Random forest":
                model = RandomForestClassifier()
                
            elif algorithm == "SVM":
                model = SVC()
            
            elif algorithm == "Decision tree":
                model = DecisionTreeClassifier()
        except:
            return "Error! No machine learning model chosen."
        # Fit data into model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        # Model evaluation metrics
        print(f"Testing {algorithm}...\n")
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        if modeMarti:
            precision = round(precision_score(y_test, y_pred, pos_label=1), 2)
        else:
            precision = round(precision_score(y_test, y_pred, pos_label='1'), 2)
        print("Precision: {:.2f}%".format(precision * 100))
        if modeMarti:
            recall = round(recall_score(y_test, y_pred, pos_label=1), 2)
        else:
            recall = round(recall_score(y_test, y_pred, pos_label='1'), 2)        
        print("Recall: {:.2f}%".format(recall * 100))
        if modeMarti:
            f1 = round(f1_score(y_test, y_pred, pos_label=1),2)
        else:
            f1 = round(f1_score(y_test, y_pred, pos_label='1'),2)
        print("F1 Score: {:.2f}%".format(f1 * 100))
        
        self.results.loc[len(self.results)] = [algorithm, accuracy, precision, recall, f1]
        
        # Visualization of the metrics
        if display_conf_matrix:
            if modeMarti:
                cm = confusion_matrix(y_test, y_pred, labels=[1,0])
            else:
                cm = confusion_matrix(y_test, y_pred, labels=['1', '2'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ilicit', 'Licit'])
            disp.plot()
            plt.title(algorithm)
            plt.show()
            
    def get_results(self):
        return self.results