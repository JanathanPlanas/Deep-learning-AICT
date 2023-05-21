
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Data_Loader import DATA_1M
# Standard PyTorch imports
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier

# Make device agnostic code


class MultiLabelClassKNN():

    def __init__(self) -> None:
        
        self.model_knn = DATA_1M(seconds=2,columns=10, jump_time =0 , n_jumps=2)

        self.Data_Loaded = self.model_knn.loading_data()  


    def Knn(self,n_neighbors =19,weights ='distance',p= 1):
        

        X_train, X_test, y_train, y_test  = self.model_knn.Spliting(data=self.Data_Loaded, 
                                                                  random_state= 42,
                                                                  test_size= 0.25,
                                                                  shuffle=True)

        self.X_train= X_train
        self.X_test= X_test
        self.y_train= y_train
        self.y_test= y_test

        

        knn = KNeighborsClassifier(n_neighbors= n_neighbors,weights=weights, p=p)

        self.model = knn.fit(self.X_train ,self.y_train)

        self.predict = self.model.predict(self.X_test)

        self.metrics = print({
            "Score": [(self.model).score(self.X_test, self.y_test)],
            "Accuracy": [accuracy_score(self.y_test, self.predict)],
            "F1-Score": [f1_score(self.y_test, self.predict, average='weighted')],
            'Recall': [recall_score(self.y_test, self.predict, average='weighted')]

        })

        if len(np.unique(y_train)) == 3:
            target_names = ["CLEAR", "LTE", "WIFI"]
        else:
            target_names = ["CLEAR", "SINAL"]
            

        self.target_names = target_names

        self.matrix = confusion_matrix(self.y_test, self.predict)

        
        print(classification_report(self.y_test, self.predict,target_names=self.target_names))

        plt.figure(figsize=(16, 7))

        sns.heatmap(
            self.matrix,
            xticklabels=self.target_names, 
            yticklabels = self.target_names,
              annot=True,
              fmt=".1f")
        
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

        

    def confusion_matrix(self) :
        

        plt.figure(figsize=(16, 7))

        sns.heatmap(
            self.matrix,
            xticklabels=self.target_names, 
            yticklabels = self.target_names,
              annot=True,
              fmt=".1f")
        
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

 
class Running():

        def All_together(self):

            print(MultiLabelClassKNN().Knn())
            print("\n-------")
            # print(MultiLabelClass().confusion_matrix())



            




if __name__== "__main__":

    Running().All_together()