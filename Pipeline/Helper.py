# Function to separate the labels of the data
def separate_labels(train_set, test_set):
    train_set = train_set.loc[train_set['class'].isin([0, 1])] # Drop unknown
    y_train = list(train_set['class'])
    X_train = train_set.drop(columns=['class'])

    test_set = test_set.loc[test_set['class'].isin([0, 1])] # Drop unknown
    y_test = list(test_set['class'])
    X_test = test_set.drop(columns=['class'])

    return X_train, y_train, X_test, y_test


# Function to separate both the labels and the classes of the dataset
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