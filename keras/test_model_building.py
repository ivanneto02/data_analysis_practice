import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
import sys

def main():

    # Create the model
    inputs = keras.Input(shape=(784,), name='digits') # 28x28 graycale images
    x = keras.layers.Dense(512, activation='relu', name='dense_1')(inputs)
    x = keras.layers.Dense(512, activation='relu', name='dense_2')(x)
    x = keras.layers.Dense(512, activation='relu', name='dense_3')(x)
    x = keras.layers.Dense(512, activation='relu', name='dense_4')(x)
    x = keras.layers.Dense(512, activation='relu', name='dense_5')(x)
    x = keras.layers.Dense(512, activation='relu', name='dense_6')(x)
    x = keras.layers.Dense(512, activation='relu', name='dense_7')(x)
    x = keras.layers.Dense(512, activation='relu', name='dense_8')(x)
    outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    
    # Define model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess data
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]

    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    my_model = Path('./saved_models/mnist_model.h5')
    if my_model.exists():
        model = keras.models.load_model('./saved_models/mnist_model.h5')
    else:
        # Compilation
        model.compile(
            # Optimizer
            optimizer = keras.optimizers.RMSprop(),

            # Loss function to minimize
            loss = keras.losses.SparseCategoricalCrossentropy(),

            # List of metrics to monitor
            metrics = [keras.metrics.SparseCategoricalAccuracy()],
        )

        # Fit the model
        history = model.fit(

            x_train,
            y_train,
            batch_size=1000,
            epochs=40,

            # We pass some validation for monitoring validation loss and metrics at the end of each epoch
            validation_data = (x_val, y_val),
        )

        # Saving the model
        model.save('./saved_models/mnist_model.h5')

    # Model evaluation
    print('Evaluating model...')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    # Prediction parameters
    pred_num = int(sys.argv[1])

    wrong = 0

    # Make prediction JUST FOR FUN
    print('\n\nGenerating predictions for 3 samples in the test dataset')
    predictions = model.predict(x_test[:pred_num])
    print('Predictions...')

    for i in range(0, len(predictions)):

        left = int(y_test[i])
        right = predictions[i].tolist().index(max(predictions[i]))

        msg = '{}. True value: {}, prediction: {}'.format(i, left, right, end='')

        if left != right:
            print(msg + ' -- WRONG')
            wrong += 1
            continue

    print('Performance: {}/{}, {}'.format(pred_num - wrong, pred_num, (pred_num - wrong)/pred_num))

if __name__ == '__main__':
    main()