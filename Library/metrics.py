import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler(feature_range=(0, 1))

class metrics:

    def __init__(self, model, X_train, X_test, y_train, y_test) -> None:

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.predict = model.predict(X_test)

    def table(self):

        return pd.DataFrame(data={
            "Score": [(self.model).score(self.X_test, self.y_test)],
            "Accuracy": [accuracy_score(self.y_test, self.predict)],
            "F1-Score": [f1_score(self.y_test, self.predict, average='weighted')],
            'Recall': [recall_score(self.y_test, self.predict, average='weighted')]

        })

    def confusion_matrix(self):

        return confusion_matrix(self.y_test, self.predict)

    def report(self):

        print(classification_report(self.y_test, self.predict))

# CONFUSION MATRIX
