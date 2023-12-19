import abc
import asyncio
from typing import Any

import numpy as np

import keras


class Polynomial(abc.ABC):

    def __init__(self):
        self.equation = None

    def get_data_matrix(self):
        pass

    def compute(self, x: np.ndarray):
        return self.equation(x)


class D2Polynomial(Polynomial):

    def __init__(self):
        super().__init__()
        self.equation = lambda x: x[0] * x[0] + 1

    def get_data_matrix(self):
        x: np.ndarray[Any, np.dtype[float]] = np.array([
            [5, 7, 8, 4]], dtype=float).reshape(2, 2)
        y: np.ndarray[Any, np.dtype[float]] = np.array([36, 33], dtype=float).reshape(2, 1)

        x_test = np.array([10.0, 20.0], dtype=float).reshape(1, 2)
        y_test = np.array([i for i in self.compute(x_test)], dtype=float)
        print(f"X -> {x.shape} y -> {y.shape}, X_test -> {x_test.shape} y_test -> {y_test.shape}")
        return x, y, x_test, y_test


class D1Polynomial(Polynomial):

    def __init__(self):
        super().__init__()
        self.equation = lambda a: 2 * a + 1

    def get_data_matrix(self):
        x = np.array([5, 7, 8, 4, 3, 23, 2, 5, 7, 9, 90, 6, 65, 5, 4, 34, 33, 3, 545, 2332], dtype=float).reshape(20, 1)
        y = np.array([i for i in self.compute(x)], dtype=float).reshape(20, 1)
        x_test = np.array([10.0, 20.0], dtype=float).reshape(2, 1)
        y_test = np.array([i for i in self.compute(x_test)], dtype=float).reshape(2, 1)
        print(f"X -> {x.shape} y -> {y.shape}, X_test -> {x_test.shape} y_test -> {y_test.shape}")
        return x, y, x_test, y_test


async def train_model(degree: int, x_train, y_train, x_test, y_test):
    match degree:
        case 1:
            model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        case 2:
            model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])
        case _:
            raise NotImplementedError(f"Polynomial degree {degree} is not supported")

    model.compile(optimizer='RMSprop', loss="mean_squared_error")
    print(model.summary())
    model.fit(x_train, y_train, epochs=5000)
    print(f"Actual output = {y_test}")
    print(f"Predicted output = {model.predict(x_test)}")


def main():
    x1, y1, x1_test, y1_test = D1Polynomial().get_data_matrix()
    # x2, y2, x2_test, y2_test = D2Polynomial().get_data_matrix()
    asyncio.run(train_model(1, x1, y1, x1_test, y1_test))
    # asyncio.run(train_model(2, x2, y2, x2_test, y2_test))


if __name__ == '__main__':
    main()
