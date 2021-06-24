from re import X
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from sklearn import datasets

import numpy as np
import random as rm

def main():

    # Load the dataset
    irises = datasets.load_iris()

    # Information about the dataset
    print('\nInformation...')
    print('target classes:', irises.target_names)
    print('iris features:', irises.feature_names, end='\n\n')

    # Make tmp for randomization purposes
    x_train_tmp = irises.data
    y_train_tmp = irises.target

    # Real final lists
    x_train = irises.data
    y_train = irises.target

    # list with new indexes
    reindex_arr = np.random.permutation(len(x_train))

    # Reindex
    for i in range(0, len(x_train)): 
        x_train[i] = x_train_tmp[reindex_arr[i]]
        y_train[i] = y_train_tmp[reindex_arr[i]]

    # Normalize
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train_normalized = (x_train - x_train_mean) / x_train_std
    x_train = x_train_normalized

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape, end='\n\n')

    # Define hyperparameters
    epochs = 50
    batch_size = 25
    validation_split = 0.2
    test_split = 0.3

    validation_size = int(len(x_train) * validation_split)
    test_size = int(len(x_train) * test_split)
    
    # Set validation data
    x_val = x_train[:validation_size]
    y_val = y_train[:validation_size]
    x_train, y_train = x_train[validation_size:], y_train[validation_size:] # Update it once to get rid of val data
    print('x_val shape:', x_val.shape)
    print('y_val shape:', y_val.shape, end='\n\n')

    # Set test data
    x_test = x_train[:test_size]
    y_test = y_train[:test_size]
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape, end='\n\n')

    # Update training data once more to get rid of test data
    x_train = x_train[test_size:]
    y_train = y_train[test_size:]

    # Make layers
    x_in = keras.Input(shape=(4,), name='iris_features')
    x = keras.layers.Dense(512, activation='relu', name='dense_1')(x_in)
    x = keras.layers.Dense(512, activation='relu', name='dense_2')(x)
    x_out = keras.layers.Dense(3, activation='softmax', name='predictions')(x)

    # Make a model
    model = keras.Model(inputs=x_in, outputs=x_out)

    # See what the model looks like
    model.summary()

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

    pred_num = len(x_test)

    # Make predictions
    print('Generating predictions...')
    predictions = model.predict(x_test[:pred_num])

    wrong = 0

    # Print all our predictions
    print('Predictions...')
    print(predictions)
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