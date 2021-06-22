import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def main():

    # Create the model
    inputs = keras.Input(shape=(784,), name='digits') # 28x28 graycale images
    x = keras.layers.Dense(100, activation=tf.nn.relu, name='dense_1')(inputs)
    x = keras.layers.Dense(100, activation=tf.nn.relu, name='dense_2')(x)
    outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    
    # Define model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    plt.imshow(x_test[int(sys.argv[1])])
    plt.show()

if __name__ == '__main__':
    main()