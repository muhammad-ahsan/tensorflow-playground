import tensorflow as tf

from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.losses import SparseCategoricalCrossentropy

from commons.TfModel import TfModel


class SequentialModel(TfModel):
    def __init__(self):
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.model = Sequential([Flatten(input_shape=(28, 28)),
                                 Dense(128, activation='relu'),
                                 Dropout(0.2),
                                 Dense(10)])

        self.model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])
        print(self.model.summary())

    def fit(self, x_train, y_train):
        self._show_baseline(x_train, y_train)
        self.model.fit(x_train, y_train, epochs=5)

    def predict(self, x_train):
        predictions = self.model(x_train[:1]).numpy()
        # The tf.nn.softmax function converts these logits to probabilities for each class:
        return tf.nn.softmax(predictions).numpy()

    def evaluate(self, x_test, y_test):
        pass

    def _show_baseline(self, x_train, y_train) -> None:
        predictions = self.model(x_train[:1]).numpy()
        print(f"Untrained model probabilities are {self.loss_fn(y_train[:1], predictions).numpy()}")


def train_test_split(mnist_dataset):
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def main() -> None:
    # Library Version
    print("TensorFlow version:", tf.__version__)
    # Split dataset
    (x_train, y_train), (x_test, y_test) = train_test_split(mnist)

    model = SequentialModel()
    # Model fitting
    model.fit(x_train, y_train)

    # Evaluate Model
    model.evaluate(x_test, y_test)

    # Get predictions
    print(model.predict(x_test))


if __name__ == '__main__':
    main()
