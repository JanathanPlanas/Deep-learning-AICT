
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler(feature_range=(0, 1))


class Normalizing:

    def __init__(self, X_train, X_test) -> None:

        self.X_train = mm.fit_transform(X_train)
        self.X_test = mm.fit_transform(X_test)
