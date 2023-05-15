
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Standard PyTorch imports
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from Library.CnnModel import TorchDataset
# Make device agnostic code

class MultiLabelClass(TorchDataset):

    def __init__(self, CLEAR, WIFI, LTE,
                  reshape : bool,
                    sliced:int, 
                    values_reshaped :int,
                    random_state, test_size, shuffle) -> None:
        
        dataset = TorchDataset(CLEAR, WIFI, LTE, reshape , sliced, values_reshaped )

        

        self.reshaped = dataset.reshaped

        self.spliting = dataset.Spliting(self.reshaped,random_state, test_size , shuffle)

    def Knn(self,n_neighbors,weights,p):

        X_train, X_test, y_train, y_test  = self.spliting
        self.X_train= X_train
        self.X_test= X_test
        self.y_train= y_train
        self.y_test= y_test
        
        knn = KNeighborsClassifier(n_neighbors= n_neighbors,weights=weights, p=p)

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
            self.y_test, self.predict),xticklabels=self.target_names, 
            yticklabels = self.target_names,
              annot=True,
              fmt=".1f")
        
        plt.xlabel('Predicted')
        plt.ylabel('Truth')


    



class Binary_Classification():

    def __init__(self, CLEAR, SINAL, 
                 reshape : bool, 
                 sliced:int, 
                 values_reshaped :int
                 ):


        """
     Initializes the TorchDatase class.
     
        """

        self.CLEAR = CLEAR.ravel()[:sliced]
        self.SINAL = SINAL.ravel()[:sliced]

        if (len(CLEAR) == len(SINAL) ) == False : # Validando que tamanho dos conjunto de dados
             
             raise ValueError("Dataset shape does not match {CLEAR.shape}")

        elif reshape == True: # Tranformando o conjunto de dados pra 3 features


            if((len(self.CLEAR) % 2 == 0) == False ):
    
                raise ValueError("Dataset size does not allow the reshape request {self.CLEAR} into shape 2")

            self.SINAL_SINAL = pd.DataFrame(np.concatenate((SINAL[:sliced].real,
                                                            SINAL[:sliced].imag),
                                                            axis=1).reshape(-1,values_reshaped)).reset_index() 
            
            self.CLEAR_SINAL = pd.DataFrame(np.concatenate((CLEAR[:sliced].real,
                                                             CLEAR[:sliced].imag),
                                                             axis=1).reshape(-1,values_reshaped)).reset_index()


            self.CLEAR_SINAL['Label'] = "CLEAR"

            self.SINAL_SINAL['Label'] = "SINAL"



            self.reshaped= pd.concat([self.CLEAR_SINAL, self.SINAL_SINAL], 
                                  keys=[
                                    f"CLEAR", f"SINAL"]).reset_index().rename(columns={'level_0': 'Signals',
                                                                        'level_1': 'index'}).drop(columns=['index'])
            
            
            self.reshaped['Label'] = (self.reshaped['Label']).map({'CLEAR':0,'SINAL':1})
        else:

            real_1, imag_1 = np.real(self.CLEAR) , np.imag(self.CLEAR)
            real_2, imag_2 = np.real(self.SINAL) , np.imag(self.SINAL)
                
                
            self.CLEAR = pd.DataFrame(real_1,
                                            imag_1).reset_index().rename(columns={'index':"Real", 0: "Imag"})
        
            self.SINAL = pd.DataFrame(real_2,
                                            imag_2).reset_index().rename(columns={'index': "Real",  0: "Imag"})
            
            self.SINAL['Label'] = "SINAL"
            self.CLEAR['Label'] = "CLEAR"


            self.dataframe= pd.concat([self.CLEAR, self.SINAL, self.LTE], keys=[
                                    f"CLEAR", f"SINAL"]).reset_index().rename(columns={'level_0': 'Signals',
                                                                        'level_1': 'index'}).drop(columns=['index'])
            
            self.dataframe['Label'] = (self.dataframe['Label']).map({'CLEAR':0,'SINAL':1})
            

    def Spliting(self,data,random_state, test_size , shuffle: bool) :


            data = data.drop(columns ='Signals' )
            X = data.iloc[:,:len(data.columns)-1]
            y = data.iloc[:,len(data.columns)-1]
                            
            X_train, X_test, y_train, y_test = train_test_split( X, y, 
                                                                test_size=test_size, 
                                                                random_state=random_state, shuffle = shuffle)
        
            
            self.X_train= X_train
            self.X_test= X_test
            self.y_train= y_train
            self.y_test= y_test
            
            return X_train, X_test, y_train, y_test

    
    def Knn(self,n_neighbors, weights, p):

        knn = KNeighborsClassifier(n_neighbors= n_neighbors,weights=weights, p= p)
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
            self.y_test, self.predict),
            xticklabels=self.target_names,
              yticklabels = self.target_names, 
              annot=True,fmt=".1f",
                cmap = sns.color_palette("light:#5A9", as_cmap=True)
)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')


        