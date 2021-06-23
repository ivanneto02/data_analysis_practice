import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
from pathlib import Path

def main():

    num_dict = {

        0: 'ZERO',
        1: 'ONE',
        2: 'TWO',
        3: 'THREE',
        4: 'FOUR',
        5: 'FIVE',
        6: 'SIX',
        7: 'SEVEN',
        8: 'EIGHT',
        9: 'NINE',
    }

    # Load the model
    my_model = './saved_models/mnist_model.h5'
    if Path(my_model).exists():
        model = keras.models.load_model(my_model)
    else:
        print('Model does not exist')
        return
    
    # Load custom image
    my_image = mpimg.imread('./img/trick.png')

    plt.imshow(my_image)
    plt.show()

    my_image = my_image[:,:,0]
    my_image = my_image.reshape(1, 784)

    prediction = model.predict(my_image)
    print(num_dict[prediction[0].tolist().index(max(prediction[0]))])

if __name__ == '__main__':
    main()