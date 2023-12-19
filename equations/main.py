import numpy as np

import keras


def polynomial_degree_one(x: np.ndarray):
    # y = 2x + 1
    return 2 * x + 1.0


def main():
    x = np.array([5, 7, 8, 4, 3, 23, 2, 5, 7, 9, 90, 6, 65, 5, 4, 34, 33, 3, 545, 2332, 45], dtype=float)
    y = np.array([i for i in polynomial_degree_one(x)], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='RMSprop', loss="mean_squared_error")
    print(model.summary())
    model.fit(x, y, epochs=5000)
    x_test = np.array([10.0, 20.0], dtype=float)
    actual = polynomial_degree_one(x_test)
    print(f"Actual output = {actual}")
    print(f"Predicted output = {model.predict(x_test)}")


if __name__ == '__main__':
    main()
