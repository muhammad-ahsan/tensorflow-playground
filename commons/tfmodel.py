from abc import ABC


class TfModel(ABC):
    def fit(self, x_train, y_train):
        pass

    def predict(self, x_train):
        pass

    def evaluate(self, x_test, y_test):
        pass
