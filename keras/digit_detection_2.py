
# Sklearn imports
from sklearn import datasets

# Keras imports
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np

def main():

    # Load database for initial analysis
    digits = datasets.load_digits()

    print(digits.data)
    print(digits.target)

    x_train = digits.data
    y_train = digits.target

    # Plot the matrices
    plot_image_matrix(digits.images, 3, 4)

    plt.show()

    # Make layers
    x_in = keras.Input(shape=(64,), name='digits')
    x = keras.layers.Dense(32, activation='relu', name='dense_1')(x_in)
    x = keras.layers.Dense(32, activation='relu', name='dense_2')(x)
    x = keras.layers.Dense(32, activation='relu', name='dense_3')(x)
    x_out = keras.layers.Dense(10, activation='softmax', name='softmax')(x)

    # Make a model
    model = keras.Model(inputs=x_in, outputs=x_out)

    model.summary()

    # Hyperparameters
    epochs = 40
    batch_size = 128
    validation_split = 0.2
    validation_size = int(len(x_train) * validation_split)
    test_split = 0.1
    test_size = int(len(x_train) * test_split)
    pred_num = 179

    # Test data
    x_test = x_train[:test_size]
    y_test = y_train[:test_size]
    x_train, y_train = x_train[test_size:], y_train[test_size:] # Update x_train and y_train
    print('test size:', len(x_test))

    # Validation data
    x_val = x_train[:validation_size]
    y_val = y_train[:validation_size]
    print('validation size:', len(x_val))

    # Final training data
    x_train, y_train = x_train[validation_size:], y_train[validation_size:]

    # Compile model
    model.compile(
        optimizer='rmsprop',

        # Loss function to minimize
        loss = keras.losses.SparseCategoricalCrossentropy(),

        # List of metrics to monitor
        metrics = [keras.metrics.SparseCategoricalAccuracy()],
    )

    # Fit model
    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (x_val, y_val),
    )

    # Make predictions
    print('Generating predictions...')
    predictions = model.predict(x_test[:pred_num])

    wrong = 0

    # Print all our predictions
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

def plot_image_matrix(images, cols, rows):
    ''' Displays the images in matrix form (9x9 grid)'''

    grid_size = cols * rows

    images = images[:grid_size]

    fig = plt.figure()
    for n, image in enumerate(images):
      a = fig.add_subplot(3, np.ceil(grid_size/float(3)), n + 1)
      plt.imshow(image, cmap='gray')
      a.set_title('img' + str(n))

    fig.tight_layout(pad=0.08)
    fig.set_size_inches(np.array(fig.get_size_inches()) * grid_size//5)

if __name__ == "__main__":
    main()