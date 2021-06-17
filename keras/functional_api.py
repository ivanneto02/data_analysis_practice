import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from keras import layers

def main():
    inputs = keras.Input(shape=(784,))
    print(inputs)

if __name__ == "__main__":
    main()