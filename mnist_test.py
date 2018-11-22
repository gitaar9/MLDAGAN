import tensorflow as tf
import numpy as np


def load_mnist_in_right_format(num_sumples):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data = [[], [], [], [], [], [], [], [], [], []]
    for idx, y in enumerate(y_train):
        # We sampled enough from every class
        if all([class_count == num_sumples for class_count in hist]):
            break
        # We sampled enough from this class
        if hist[y] == num_sumples:
            continue
        # Otherwise add the sample to the class and increase the classes count
        data[y].append([[[col] for col in row] for row in x_train[idx]])
        hist[y] += 1
    return np.array(data)


data = load_mnist_in_right_format(100)
print(data.shape)
for class_data in data:
    print(class_data.shape)
