import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    example1()
    example2()
    example3()
    example4()
    example5()
    example6()

def example1():
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

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
    x_test = x_test.reshape(10000, 784).astype("float32") / 255.0

    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics = ["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)

    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save('./saved_models/my_first_model')

    del model

    model = keras.models.load_model('./saved_models/my_first_model')

    model.summary()

def example2():

    encoder_input = keras.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(16, 2, activation='relu')(encoder_input)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name="encoder")
    encoder.summary()

    x = layers.Reshape((4, 4, 1))(encoder_output)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)

    autoencoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name="autoencoder")
    autoencoder.summary()

def example3():

    ```Encoder```
    # Input layer
    encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")

    # Hidden layers
    x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    
    # Output layer
    encoder_output = layers.GlobalMaxPooling2D()(x)

    # Create model and summarize it
    encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
    encoder.summary()

    ```Decoder```
    # Input layer 
    decoder_input = keras.Input(shape=(16,), name='encoded_img')

    # Hidden layers
    x = layers.Reshape((4, 4, 1))(decoder_input)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)

    # Output layer
    decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

    # Create model and summarize it
    decoder = keras.Model(inputs = decoder_input, outputs = decoder_output, name = 'decoder')
    decoder.summary()

def example4():
    pass

def example5():
    pass

def example6():
    pass

if __name__ == "__main__":
    main()