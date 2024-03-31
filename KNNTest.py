"""Provides tools for building and testing kNN models based on given data. 
There are also tools allowing to make correlation tests. 
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import scipy
import itertools

class KNN_model():
    """Build KNN model"""
    def __init__(self, df, class_list, features_list):
        """
        Keyword arguments:
        df -- processed pandas data frame
        class_list -- list containing class names
        features_list -- list containing feature names
        """
        self.df = df
        self.class_list = class_list
        self.features_list = features_list

    def build_model(self, class_col, size=0.3, n=3):
        """Build a classification model

        Keyword arguments:
        class_col -- Column containing class names of the model
        size -- test_size value in train_test_split function
        n -- n_neighbours value in KNeighborsClassifier constructor
        """
        X = np.array([self.df[feature] for feature in self.features_list])
        X = X.transpose()
        self.X = X
        y = self.df[class_col].squeeze()
        self.feature_train, self.feature_test, self.target_train, self.target_test = train_test_split(X, y, test_size=size, random_state=1)
        self.model = KNeighborsClassifier(n_neighbors=n)
        self.model.fit(self.feature_train, self.target_train)
        self.predictions = self.model.predict(self.feature_test)
    
class KNN_test(KNN_model):
    """Expand KNN_model class.
    Provide tools for testing the kNN model
    """
    def test_model(self):
        """Test kNN model using confusion matrix method"""
        test_l = []
        conf_matrix = confusion_matrix(self.target_test, self.predictions, labels=self.model.classes_)
        num_classes = self.model.classes_
        for i in range(num_classes.shape[0]):
            PP = conf_matrix[i, i] #true positive
            FN = np.sum(conf_matrix[i, :]) - PP #false negative
            FP = np.sum(conf_matrix[:, i]) - PP #false positive
            PN = np.sum(conf_matrix) - PP - FN - FP #true negative
            precision = PP / (PP + FP)
            recall = PP / (PP + FN)
            f1 = (2 * precision * recall) / (precision + recall)
            test_l.append([num_classes[i], PP, PN, FP, FN, precision, recall, f1])
        return pd.DataFrame(test_l, columns=['Klasa', 'PP', 'PN', 'FP', 'FN', 'Precision', 'Recall', 'F1'])
    
    def polt_conf_matrix(self, rotate_x=0):
        conf_matrix = confusion_matrix(self.target_test, self.predictions, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.model.classes_)
        disp.plot(xticks_rotation=rotate_x)
        disp.ax_.set(xlabel="Predicted class", ylabel="Real class")
        return disp

    def test_model2(self):
        """Perform aggregated tests on the model"""
        acc = accuracy_score(self.target_test, self.predictions)
        prec = precision_score(self.target_test, self.predictions, average='weighted')
        rec = recall_score(self.target_test, self.predictions, average='weighted')
        f1 = f1_score(self.target_test, self.predictions, average='weighted')
        return [acc, prec, rec, f1]

class Correlation_test():
    """Perform correlation tests"""
    def __init__(self, f_list, df, corr_val):
        """
        Keyword arguments:
        f_list -- list containing feature names
        df -- processed pandas data frame
        corr_val -- value of minimal correlation level
        """
        self.f_list = f_list
        self.df = df
        self.combinations = self.comb()
        self.corr_l = self.corr_list()
        self.cut_l = self.corr_cut(corr_val)

    def comb(self):
        """Generate combinations of different list lengths"""
        combinations = []
        for r in range(1, len(self.f_list) + 1):
            combinations.extend(list(itertools.combinations(self.f_list, r)))
        return combinations

    def cor_result(self, f1, f2):
        """Calculate correlation level
        
        Keyword arguments:
        f1 -- series object containing values of first feature
        f2 -- series object containing values of second feature
        """
        cor = scipy.stats.pearsonr(f1, f2)
        cor = np.round(cor, decimals=5)
        return cor

    def corr_list(self):
        """Generate list with different features combinations 
        and thier correlation level value
        """
        temp_l = self.f_list[1:]
        corr_l = []
        for f in self.f_list:
            for tl in temp_l:
                result = self.cor_result(self.df[f], self.df[tl])
                corr_l.append([f, tl, result[0], result[1]])
            temp_l = temp_l[1:]
        return corr_l

    def corr_cut(self, val):
        """Return list containing combinations of features
        excluding features with high correlation level
        
        Keyword argument:
        val -- value of minimal correlation level
        """
        corr_df = pd.DataFrame(self.corr_l, columns=['Feature 1', 'Feature 2', 'Correlation level', 'p'])
        corr_l = [[a[0], a[1]] for a in corr_df[(abs(corr_df['Correlation level']) >= val)].values]
        for l in corr_l:
            for i in self.combinations:
                if l[0] in i and l[1] in i:
                    #remove features with high correlation level:
                    self.combinations.remove(i)
        return [list(c) for c in self.combinations]
    
class Data_iter(KNN_test, Correlation_test):
    """Iterate through data with different k-values and features combinations"""
    def __init__(self, df, class_list, features_list, corr_val, class_col):
        """
        Keyword arguments:
        df -- processed pandas data frame
        class_list -- list containing class names
        features_list -- list containing feature names
        corr_val -- value of minimal correlation level
        """
        self.df = df = df[df[class_col].isin(class_list)]
        self.features_list = features_list
        self.class_col = class_col
        self.combinations = self.comb()
        self.corr_l = self.corr_list()
        self.cut_l = self.corr_cut(corr_val)

    def test_iter(self, k_values):
        """Build models with different k-values and features combinations
        
        Keyword argument:
        k_values -- list containing values of the k-parameter
        """
        rate_l =[]
        #remove from list combinations with only one feature:
        f_l = [l for l in self.cut_l if len(l) > 1]
        for k in k_values:
            for f in f_l:
                self.features_list = f
                self.build_model(self.class_col, 0.3, k)
                rate_l.append([k, f] + self.test_model2())
        return rate_l

    def test_iter2(self, k_values):
        """Build models with different k-values
        
        Keyword argument:
        k_values -- list containing values of the k-parameter
        """
        rate_l =[]
        for k in k_values:
            self.build_model(self.class_col, 0.3, k)
            rate_l.append([k] + self.test_model2())

        return rate_l

