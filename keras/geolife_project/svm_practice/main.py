from scipy.sparse.construct import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # test_model()
    plot_mnist()
    pass

def test_model():

    # create_model()
    my_model = load_model()

    # load dataset
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape the dataset
    x_test = x_test.reshape((10000, 784))
    y_test = y_test.reshape((10000,))

    # Make predictions
    predictions = my_model.predict(x_test)

    wrong = 0
    for pred_i in range(len(predictions)):
        if (predictions[pred_i] != y_test[pred_i]):
            wrong += 1
            print('prediction:', predictions[pred_i], '  real value:', y_test[pred_i])
    print('Accuracy:', str((len(x_test) - wrong)/len(x_test) * 100) + '%')

def load_model():

    return joblib.load('./mnist_final_model.sav')

def plot_mnist():
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape((60000, 784))

    print('Shape:', x_train.shape)
    print('Max val:', np.max(x_train))
    print('labels:', np.unique(y_train))

    nsamples = 60000
    data = x_train[:nsamples]
    labels = y_train[:nsamples]

    data = data / 255.0
    labels = labels.astype('int')

    # # With TSNE
    tsne = TSNE(n_components=2, random_state=123, verbose=1).fit_transform(data)
    print(tsne.shape)

    # Make clusters
    clustering = DBSCAN(eps=0.5, min_samples=1300, algorithm='ball_tree').fit(tsne)
    print('clustered labels:', clustering.labels_)
    print('length:', len(clustering.labels_))
    print('unique:', np.unique(clustering.labels_))

    # # Print TSNE scatter
    plt.scatter(x=tsne[:,0], y=tsne[:,1], c=labels, alpha=0.5)
    plt.title('with TSNE')
    plt.show()

    # # With PCA
    # pca = decomposition.PCA(n_components=2)
    # view = pca.fit_transform(data)

    # # Make clusters
    # clustering = DBSCAN(eps=0.5, min_samples=1300, algorithm='ball_tree').fit(view)
    # print('clustered labels:', clustering.labels_)
    # print('length:', len(clustering.labels_))
    # print('unique:', np.unique(clustering.labels_))
    
    # # Plot PCA scatter
    # plt.scatter(view[:,0], view[:,1], c=labels, alpha=0.2, cmap='Set1')
    # plt.title('with PCA')
    # plt.show()

    # # With TSNE
    # tsne = TSNE(n_components=2, random_state=123).fit_transform(data)
    # plt.scatter(x=tsne[:,0], y=tsne[:,1], c=labels, alpha=0.5)
    # plt.title('with TSNE')
    # plt.show()

def create_model():

    # Import MNIST datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Print the shape to see how it looks like
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    # Reshape data
    x_train = x_train.reshape((60000, 784))
    y_train = y_train.reshape((60000,))

    # Normalize
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std

    # Normalize test
    x_test_mean = x_test.mean()
    x_test_std = x_test.std()
    x_test = (x_test - x_test_mean) / x_test_std

    # SVM with linear kernel
    CLF = SVC(kernel='linear', verbose=1)
    CLF.fit(x_train, y_train)

    # Save model to disk
    joblib.dump(CLF, './mnist_final_model.sav')

if __name__ == '__main__':
    main()