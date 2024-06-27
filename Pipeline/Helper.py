import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.metrics import *
import pandas as pd

'''
Function to merge the timesteps into train and test sets.

df_dict is a dictionary where each element is a timestep of the dataset
method can be sequential (timesteps sorted from 1 to 49) or balanced (timesteps sorted by amount of illicit nodes)
section is a list
'''
def merge_timesteps(df_dict, method, section):
    train_df = pd.DataFrame()         
    test_df = pd.DataFrame()  

    if method == 'sequential':
        for i in range(section[0], section[1]+1):
            train_df = pd.concat([train_df, df_dict[i]], ignore_index=True)

        for ii in range(section[1]+1, section[2]+1):
            test_df = pd.concat([test_df, df_dict[ii]], ignore_index=True) 
    
    elif method == 'balanced':
        ilicit_count = []
        # Sort the time steps given their amount of illicit nodes
        for key in df_dict.keys():        
            temp = df_dict[key].groupby('class').count()   
            temp = temp['node'].reset_index()
            ilicit_count.append([key, temp[temp['class'] == 0]['node'][0]])
        ilicit_count.sort(key = lambda row: row[1], reverse=True) 

        for i in range(0, min(section[0]*2, 50), 2):
            train_df = pd.concat([train_df, df_dict[ilicit_count[i][0]]], ignore_index=True)
        
        for ii in range(1, min(section[1]*2, 50), 2):
            test_df = pd.concat([test_df, df_dict[ilicit_count[ii][0]]], ignore_index=True)

        while (len(train_df['time step'].unique()) < section[0]):
            ii+=2                       
            train_df = pd.concat([train_df, df_dict[ilicit_count[ii][0]]], ignore_index=True) 
        
        while (len(test_df['time step'].unique()) < section[1]):
            i+=2                       
            test_df = pd.concat([test_df, df_dict[ilicit_count[i][0]]], ignore_index=True) 
    
    return train_df, test_df


'''
Function to separate the labels of the data, dropping the unknown nodes. For training.
'''
def separate_labels(train_set, test_set):
    train_set = train_set.loc[train_set['class'].isin([0, 1])] # Drop unknown
    y_train = list(train_set['class'])
    X_train = train_set.drop(columns=['class'])

    test_set = test_set.loc[test_set['class'].isin([0, 1])] # Drop unknown
    y_test = list(test_set['class'])
    X_test = test_set.drop(columns=['class'])

    return X_train, y_train, X_test, y_test


'''
Function to separate both the labels and the classes of the data, also dropping the unknowns.
Useful for training the reconstruction error for example, where you need to distinguish by class.
'''
def split_labels_classes(train_set, test_set):
    # Split train data in licit and ilicit
    X_ilicit = train_set.loc[train_set['class'].isin([0])] 
    X_licit = train_set.loc[train_set['class'].isin([1])] 
    X_ilicit = X_ilicit.drop(columns=['class'])
    X_licit = X_licit.drop(columns=['class'])

    y_ilicit = list(train_set.loc[train_set['class'].isin([0])]['class'])
    y_licit = list(train_set.loc[train_set['class'].isin([1])]['class'])
    

    test_ilicit_X = test_set.loc[test_set['class'].isin([0])] 
    test_licit_X = test_set.loc[test_set['class'].isin([1])] 
    test_ilicit_X = test_ilicit_X.drop(columns=['class'])
    test_licit_X = test_licit_X.drop(columns=['class'])

    test_ilicit_y = list(test_set.loc[test_set['class'].isin([0])]['class'])
    test_licit_y = list(test_set.loc[test_set['class'].isin([1])]['class'])

    return X_ilicit, y_ilicit, X_licit, y_licit, test_ilicit_X, test_ilicit_y, test_licit_X, test_licit_y 


'''
Function to display the performance of the Machine Learning classifiers
'''
def plot(name, y_test, y_pred, save_results=True, df_results=None, CM=False):    
    # Except for the accuracy, the others compute the metric for the ilicit class
    print(f"Testing {name}...")
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    precision = round(precision_score(y_test, y_pred, pos_label=0), 4)
    print("Precision: {:.2f}%".format(precision * 100))
    recall = round(recall_score(y_test, y_pred, pos_label=0), 4)
    print("Recall: {:.2f}%".format(recall * 100))
    f1 = round(f1_score(y_test, y_pred, pos_label=0),4)
    print("F1 Score: {:.2f}%\n".format(f1 * 100))
    
    if save_results:
        df_results = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1 score"])
        if "DecisionTreeClassifier()" in str(name):
            df_results.loc[len(df_results)] = ["Decision Tree", accuracy, precision, recall, f1]
        
        elif "RandomForestClassifier()" in str(name):
            df_results.loc[len(df_results)] = ["Random Forest", accuracy, precision, recall, f1]
            
        elif "GradientBoostingClassifier()" in str(name):
            df_results.loc[len(df_results)] = ["Gradient Boosting", accuracy, precision, recall, f1]
        
        elif "ExtraTreesClassifier()" in str(name):
            df_results.loc[len(df_results)] = ["Extra Trees", accuracy, precision, recall, f1]
        
        else:
            df_results.loc[len(df_results)] = ["Multi-Layer Perceptron (MLP)", accuracy, precision, recall, f1]

    # Confusion matrix
    if CM:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ilicit', 'Licit'])
        disp.plot()
        plt.title(name)
        plt.show()
        
    return df_results