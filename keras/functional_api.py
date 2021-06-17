import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    inputs = keras.Input(shape=(784,))

    #img_inputs = keras.Input(shape = (32, 32, 3))

    #print(inputs.shape, inputs.dtype)
    #print(img_inputs.shape, img_inputs.dtype)

    dense = layers.Dense(64, activation='relu')

    # Layer 1
    x = dense(inputs)

    # Layer 2
    x = layers.Dense(64, activation='relu')(x)

    # Output layer
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    model.summary()

    keras.utils.plot_model(model, "./model_images/my_first_model.png")
    keras.utils.plot_model(model, "./model_images/my_first_model_with_shape_info.png", show_shapes=True)

if __name__ == "__main__":
    main()