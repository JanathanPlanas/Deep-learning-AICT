
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

    def __init__(self, data , 
                    random_state, test_size, shuffle) -> None:
        
        dataset = TorchDataset()

        self.spliting = dataset.Spliting(data,random_state, test_size , shuffle)


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

        if len(np.nunique(y_train)) == 3:
            target_names = ["CLEAR", "LTE", "WIFI"]
        else:
            target_names = ["CLEAR", "SINAL"]
            

        self.target_names = target_names

        self.matrix = confusion_matrix(self.y_test, self.predict)

        self.report = print(classification_report(self.y_test, self.predict,target_names=self.target_names))

        

    def confusion_matrix(self) -> None:


        self.predicted = (self.model).predict(self.X_test)

        plt.figure(figsize=(16, 7))

        sns.heatmap(confusion_matrix(
            self.y_test, self.predict),xticklabels=self.target_names, 
            yticklabels = self.target_names,
              annot=True,
              fmt=".1f")
        
        plt.xlabel('Predicted')
        plt.ylabel('Truth')


        

if __name__== "__main__":

        def get_data():
                
            data= ['CLEAR_40SEC.npy','LTE_1M_40SEC.npy','WIFI1M_40SEC.npy']
            signal = []
            i = 0
            for i in range(len(data)):
                signal.append(np.load(f"Data_\{data[i]}"))

            return signal

        df = MultiLabelClass(get_data()[0],get_data()[1] ,get_data()[2],
               reshape=True,
                 sliced=60000,
                   values_reshaped= 10,
                   random_state= 42, 
                   test_size = 0.275, shuffle = True) ; data = df.reshaped 

        print(data)