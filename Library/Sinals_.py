
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Standard PyTorch imports
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

# Make device agnostic code

class Knn_apply():

    def __init__(self, CLEAR, WIFI, LTE, sliced,test_size,
                 shuffle :bool,random_state:int,n_neighbors, weights:str
                 , p):

        """
        Essa função vai unir os 3 dataframes (numpy.array) e criar Labels definidaas para cada input.

        
        3 dataframes (numpy.array)
        sliced : int or floar - valor limitador do tamanho do arquivo

        output : DataFrame concatenados
        """
        self.p = p 
        self.weights = weights
        self.n_neighbors = n_neighbors
        self.database1 = CLEAR.ravel()
        self.database2 = WIFI.ravel()
        self.database3 = LTE.ravel()
        self.sliced = sliced
        self.test_size = test_size
        self.shuffle = shuffle 
        self.random_state = random_state

        real_1, imag_1 = np.real(self.database1[:sliced]) , np.imag(self.database1[:sliced])
        real_2, imag_2 = np.real(self.database2[:sliced]) , np.imag(self.database2[:sliced])
        real_3, imag_3 = np.real(self.database3[:sliced]) , np.imag(self.database3[:sliced])
            


            
        self.database_1 = pd.DataFrame(real_1,
                                        imag_1).reset_index().rename(columns={'index':"Real",
                                                                                            0: "Imag"})
        self.database_1['Label'] = f"CLEAR"

    
        self.database_2 = pd.DataFrame(real_2,
                                        imag_2).reset_index().rename(columns={'index': "Real",
                                                                                             0: "Imag"})
        self.database_2['Label'] = f"WIFI"


        self.database_3 = pd.DataFrame(real_3,
                                        imag_3).reset_index().rename(columns={'index': "Real",
                                                                                            0: "Imag"})
        self.database_3['Label'] = f"LTE"


        self.dataframe= pd.concat([self.database_1, self.database_2, self.database_3], keys=[
                                f"CLEAR", f"WIFI", f"LTE"]).reset_index().rename(columns={'level_0': 'Signals',
                                                                     'level_1': 'index'}).drop(columns=['index'])
        
        self.dataframe['Label'] = (self.dataframe['Label']).map({'CLEAR':0,'WIFI':1, "LTE":2})
        

    @property
    def sliced(self):  # Picking a number wich means the captured signal
        return self._sliced

    @sliced.setter
    def sliced(self, value):
        if isinstance(value, int):
            self._sliced = value
        else:
            raise Exception("Valor fornecido não é dtype Int")
        
    

    def Training(self,data) :

        X = data[['Real','Imag']]
        y = data[['Label']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle = self.shuffle)
    
        
        self.X_train= X_train
        self.X_test= X_test
        self.y_train= y_train
        self.y_test= y_test

        return X_train, X_test, y_train, y_test
        
    def Knn(self):
        knn = KNeighborsClassifier(n_neighbors= self.n_neighbors,weights=self.weights, p=self.p)
        self.model = knn.fit(self.X_train ,self.y_train)

        self.predict = self.model.predict(self.X_test)

        self.metrics = pd.DataFrame(data={
            "Score": [(self.model).score(self.X_test, self.y_test)],
            "Accuracy": [accuracy_score(self.y_test, self.predict)],
            "F1-Score": [f1_score(self.y_test, self.predict, average='weighted')],
            'Recall': [recall_score(self.y_test, self.predict, average='weighted')]

        })
        self.target_names= ["CLEAR", "WIFI","LTE"]
        self.matrix = confusion_matrix(self.y_test, self.predict)

        self.report = print(classification_report(self.y_test, self.predict,target_names=self.target_names))

        


    def confusion_matrix(self) -> None:


        self.predicted = (self.model).predict(self.X_test)

        plt.figure(figsize=(16, 5))
        sns.heatmap(confusion_matrix(
            self.y_test, self.predict),xticklabels=self.target_names, yticklabels = self.target_names, annot=True,fmt=".1f")
        plt.xlabel('Predicted')
        plt.ylabel('Truth')


    



class merge_two():

    def __init__(self, CLEAR,SINAL, sliced,test_size,
                 shuffle :bool,random_state:int,n_neighbors, weights:str
                 , p):

        """
        Essa função vai unir os 3 dataframes (numpy.array) e criar Labels definidaas para cada input.

        
        3 dataframes (numpy.array)
        sliced : int or floar - valor limitador do tamanho do arquivo

        output : DataFrame concatenados
        """
        self.p = p 
        self.weights = weights
        self.n_neighbors = n_neighbors
        self.database1 = CLEAR.ravel()
        self.database2 = SINAL.ravel()
        self.sliced = sliced
        self.test_size = test_size
        self.shuffle = shuffle 
        self.random_state = random_state

        real_1, imag_1 = np.real(self.database1[:sliced]) , np.imag(self.database1[:sliced])
        real_2, imag_2 = np.real(self.database2[:sliced]) , np.imag(self.database2[:sliced])
            


            
        self.database_1 = pd.DataFrame(real_1,
                                        imag_1).reset_index().rename(columns={'index':"Real",
                                                                                            0: "Imag"})
        self.database_1['Label'] = f"CLEAR"

    
        self.database_2 = pd.DataFrame(real_2,
                                        imag_2).reset_index().rename(columns={'index': "Real",
                                                                                             0: "Imag"})
        self.database_2['Label'] = f"SINAL"



        self.dataframe= pd.concat([self.database_1, self.database_2], keys=[
                                f"CLEAR", f"SINAL"]).reset_index().rename(columns={'level_0': 'Signals',
                                                                     'level_1': 'index'}).drop(columns=['index'])
        
        self.dataframe['Label'] = (self.dataframe['Label']).map({'CLEAR':0,'SINAL':1})
        

    @property
    def sliced(self):  # Picking a number wich means the captured signal
        return self._sliced

    @sliced.setter
    def sliced(self, value):
        if isinstance(value, int):
            self._sliced = value
        else:
            raise Exception("Valor fornecido não é dtype Int")
        
    

    def Training(self,data) :

        X = data[['Real','Imag']]
        y = data[['Label']]
                         
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle = self.shuffle)
    
        
        self.X_train= X_train
        self.X_test= X_test
        self.y_train= y_train
        self.y_test= y_test
        
        return X_train, X_test, y_train, y_test
    
    def Knn(self):

        knn = KNeighborsClassifier(n_neighbors= self.n_neighbors,weights=self.weights, p=self.p)
        self.model = knn.fit(self.X_train ,self.y_train)

        self.predict = self.model.predict(self.X_test)

        self.metrics = pd.DataFrame(data={
            "Score": [(self.model).score(self.X_test, self.y_test)],
            "Accuracy": [accuracy_score(self.y_test, self.predict)],
            "F1-Score": [f1_score(self.y_test, self.predict, average='weighted')],
            'Recall': [recall_score(self.y_test, self.predict, average='weighted')]

        })
        self.target_names= ["CLEAR", "SINAL"]
        self.matrix = confusion_matrix(self.y_test, self.predict)

        self.report = print(classification_report(self.y_test, self.predict,target_names=self.target_names))     
        

    def confusion_matrix(self) -> None:


        self.predicted = (self.model).predict(self.X_test)

        plt.figure(figsize=(16, 5))
        sns.heatmap(confusion_matrix(
            self.y_test, self.predict),xticklabels=self.target_names, yticklabels = self.target_names, annot=True,fmt=".1f")
        plt.xlabel('Predicted')
        plt.ylabel('Truth')


    