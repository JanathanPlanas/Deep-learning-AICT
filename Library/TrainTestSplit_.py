
import numpy as np
from sklearn.model_selection import train_test_split


class TrainTestSplit:

    def __init__(self, data):
        self.data = data

    def train_test_split(self, test_size, random_state):
        X = self.data[['Real', 'Imag']]
        y = self.data['ID']
        self.test_size = test_size
        self.random_state = random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test


