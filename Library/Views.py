import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


class Visualization:
    def __init__(self):
        pass

    def plot_pie(self, y_test):
        counts = y_test.value_counts(normalize=True)
        counts.plot.pie(autopct="%0.2f%%")
        plt.show()

    def confusion_matrix(self, model, X_train, X_test, y_train, y_test) -> None:

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.predict = (model).predict(X_test)

        plt.figure(figsize=(16, 5))
        sns.heatmap(confusion_matrix(
            self.y_test, self.predict), annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')


class RandomizedSearchCVWrapper:

    def __init__(self, model, param_distributions, X_train, y_train,
                 n_iter, cv, random_state, n_jobs,
                 scoring, verbose, return_train_score) -> None:

        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.verbose = verbose
        self.return_train_score = return_train_score

        self.randomsearch = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv,
                                               random_state=random_state, n_jobs=n_jobs, scoring=scoring,
                                               verbose=verbose, return_train_score=return_train_score)

        self.randomsearch.fit(X_train, y_train)
        self.results = pd.DataFrame(self.randomsearch.cv_results_).sort_values(
            "mean_test_score", ascending=False)
        self.best_params = self.randomsearch.best_params_

    def plot_mean_performance(self):

        plt.figure(figsize=(16, 4))
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["mean_test_score"]), label="Testing Score", linestyle='dotted')
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["mean_train_score"]), label="Training Score", linestyle='dotted')
        plt.xlabel("Number of N neighbors ")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.title("Performance vs Number of K")

    def plot_std_performance(self):

        plt.figure(figsize=(16, 4))
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["std_test_score"]), label="Testing Error", linestyle='dotted')
        plt.plot(self.results["param_n_neighbors"], np.abs(
            self.results["std_train_score"]), label="Training Error", linestyle='dotted')
        plt.xlabel("Number of N neighbors ")
        plt.ylabel("Root from the Mean Absolute Error")
        plt.legend()
        plt.title("Performance vs Number of K")
