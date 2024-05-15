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


class D2Polynomial(Polynomial):

    def __init__(self):
        super().__init__()
        self.equation = lambda x, y: x * y + 1

    def compute(self, x: np.ndarray):
        results = []
        for i, x_instance in enumerate(x):
            results.append(self.equation(x[i][0], x[i][1]))
        return results

    def get_data_matrix(self):
        rows: int = 12
        columns: int = 2
        x: np.ndarray[Any, np.dtype[float]] = np.array([
            [5, 7, 8, 4, 3, 5, 34, 56, 35, 67, 456, 2343, 3423, 34343, 43, 346, 121, 44, 7865, 1352, 4523, 54545, 232,
             237]], dtype=float).reshape(rows, columns)
        y = np.array([self.compute(x)], dtype=float).reshape(rows, 1)

        num_test_examples: int = 1
        x_test = np.array([10.0, 20.0], dtype=float).reshape(num_test_examples, columns)
        y_test = np.array(self.compute(x_test), dtype=float).reshape(num_test_examples, 1)
        print(f"X -> {x.shape} y -> {y.shape}, X_test -> {x_test.shape} y_test -> {y_test.shape}")
        return x, y, x_test, y_test


class D1Polynomial(Polynomial):

    def __init__(self):
        super().__init__()
        self.equation = lambda a: 2 * a + 1

    def compute(self, x: np.ndarray):
        return self.equation(x)

    def get_data_matrix(self):
        rows: int = 20
        columns: int = 1
        num_test_examples: int = 1
        x = np.array([5, 7, 8, 4, 3, 23, 2, 5, 7, 9, 90, 6, 65, 5, 4, 34, 33, 3, 545, 2332], dtype=float).reshape(rows, columns)
        y = np.array([self.compute(x)], dtype=float).reshape(rows, 1)
        x_test = np.array([10.0], dtype=float).reshape(num_test_examples, columns)
        y_test = np.array([i for i in self.compute(x_test)], dtype=float).reshape(num_test_examples, 1)
        print(f"X -> {x.shape} y -> {y.shape}, X_test -> {x_test.shape} y_test -> {y_test.shape}")
        return x, y, x_test, y_test


async def train_model(degree: int, x_train, y_train, x_test, y_test):
    match degree:
        case 1:
            model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        case 2:
            model = keras.Sequential([keras.layers.Dense(units=1, activation='elu', input_shape=[2])])
        case _:
            raise NotImplementedError(f"Polynomial degree {degree} is not supported")

    model.compile(optimizer='RMSprop', loss="mae")
    print(model.summary())
    model.fit(x_train, y_train, epochs=5000)
    predicted = model.predict(x_test)
    assert len(predicted) == len(y_test)
    print(f"Actual output for polynomial degree {degree} = {y_test}")
    print(f"Predicted output for polynomial degree {degree} = {predicted}")


def main():
    x1, y1, x1_test, y1_test = D1Polynomial().get_data_matrix()
    x2, y2, x2_test, y2_test = D2Polynomial().get_data_matrix()
    asyncio.run(train_model(1, x1, y1, x1_test, y1_test))
    asyncio.run(train_model(2, x2, y2, x2_test, y2_test))


if __name__ == '__main__':
    main()
