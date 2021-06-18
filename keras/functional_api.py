import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    #example1()
    #example2()
    #example3()
    #example4()
    #example5()
    #example6()
    #example7()
    #example8()
    #example9()
    #example10()
    customRNNexample()

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
    decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")
    autoencoder.summary()

def example3():

    '''Encoder'''
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

    '''Decoder'''
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

    autoencoder_input = keras.Input(shape(28, 28, 1), name='img')
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = keras.Model(inputs = autoencoder_input, outputs = decoded_img, name='autoencoder')
    autoencoder.summary()


'''Ensembling a model - common usecase for model nesting'''
def example4():
    
    def get_model():
        inputs = keras.Input(shape=(128,))
        outputs = layers.Dense(1)(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    model1 = get_model()
    model2 = get_model()
    model3 = get_model()

    inputs = keras.Input(shape=(128,))
    y1 = model1(inputs)
    y2 = model2(inputs)
    y3 = model3(inputs)
    outputs = layers.average([y1, y2, y3])
    ensemble_model = keras.Model(inputs=inputs, outputs=outputs)

    ensemble_model.summary()
    keras.utils.plot_model(ensemble_model, "./model_images/ensemble_model_plot.jpg", show_shapes=True)

'''
Multiple inputs and outputs

In this example, we are categorizing tickets that
belong to one of four different departments.
'''
def example5():

    '''Initial parameters'''
    num_tags = 12       # Number of unique issue tags
    num_words = 1000    # Size of vocabulary obtained when preprocessing text data
    num_departments = 4 # Number of departments for prediction

    '''Create the model'''
    title_input = keras.Input(
        shape=(None,), name='title'
    ) # Variable-length sequence of ints

    body_input = keras.Input(
        shape=(None,), name='body'
    ) # Variable-length sequence of ints

    tags_input = keras.Input(
        shape=(num_tags,), name='tags'
    ) # Binary vectors of size `num_tags`

    # Embed each word in the title into a 64-dimensional vector
    title_features = layers.Embedding(num_words, 64)(title_input)

    # Embed each word in the text into a 64-dimensional vector
    body_features = layers.Embedding(num_words, 64)(body_input)

    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    title_features = layers.LSTM(128)(title_features)

    # Reduce sequence of embedded words in the body into a single 32-dimensional vector
    body_features = layers.LSTM(32)(body_features)

    # Merge all available features into a single layer vector via concatenation
    x = layers.concatenate([title_features, body_features, tags_input])

    # Stick a logistic regression for priority prediction on top of the features
    priority_pred = layers.Dense(1, name='priority')(x)

    # Stick a department classifier on top of the features
    department_pred = layers.Dense(num_departments, name='department')(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs = [title_input, body_input, tags_input],
        outputs = [priority_pred, department_pred],
    )

    '''Plot the model'''
    keras.utils.plot_model(model, './model_images/multi_input_and_output_model.png', show_shapes=True)

    model.summary()

    '''Compile the model - example1'''
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[
            keras.losses.BinaryCrossentropy(from_logits=True),
            keras.losses.CategoricalCrossentropy(from_logits=True),
        ],
        loss_weights=[1.0, 0.2],
    )

    '''Compile the model - example2'''
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={
            'priority' : keras.losses.BinaryCrossentropy(from_logits=True),
            'department' : keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights = {'priority' : 1.0, 'department' : 0.2},
    )
    
    # Dummy input data
    title_data = np.random.randint(num_words, size=(1280, 10))
    body_data = np.random.randint(num_words, size=(1280, 100))
    tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')

    # Dummy target data
    priority_targets = np.random.random(size=(1280, 1))
    department_targets = np.random.randint(2, size=(1280, num_departments))

    # Parameters
    epochs = 10
    batch_size = 32

    '''Fit the model'''
    model.fit(
        {'title' : title_data, 'body' : body_data, 'tags' : tags_data},
        {'priority' : priority_targets, 'department' : department_targets},
        epochs = epochs,
        batch_size = batch_size,
    )

def example6():
    inputs = keras.Input(shape=(32, 32, 3), name='img')
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)

    '''Make the model and plot it'''
    model = keras.Model(inputs=inputs, outputs=outputs, name='toy_resnet')
    model.summary()
    keras.utils.plot_model(model, './model_images/mini_resnet.png', show_shapes=True)

    '''Train the model'''

    # Get the raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Compile the model
    model.compile(
        optimizer = keras.optimizers.RMSprop(1e-3),
        loss = keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
    )

    # Parameters
    batch_size = 64
    epochs = 1
    validation_split = 0.2

    # We restrict the data to the first 1000 samples so as to limit execution time
    # on Colab. Try to train on the entire dataset until convergence!
    model.fit(
        x_train[:1000],
        y_train[:1000],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )

'''
Shared layers

Shared layers are layers that are reused multiple times in the same model
They learn features that correspond to multiple paths in the graph-of-layers
'''
def example7():
    # Embedding for 1000 unique words mapped to 128-dimensional vectors
    shared_embedding = layers.Embedding(1000, 128)

    # Variable-length sequence of integers
    text_input_a = keras.Input(shape=(None,), dtype='int32')

    # Variable-length sequence of integers
    text_input_b = keras.Input(shape=(None,), dtype='int32')

    # Reuse the same layer to encode both inputs
    encoded_input_a = shared_embedding(text_input_a)
    encoded_input_b = shared_embedding(text_input_b)

'''
Extract and reuse nodes in the graph of layers

Example for accessing activations of itermediate layers, which can
be useful for something like feature extractions. You can access the
nodes in the graph and reuse them elsewhere.
'''
def example8():
    vgg19 = tf.keras.applications.VGG19()

    # Access intermediate layers
    features_list = [layer.output for layer in vgg19.layers]

    # Use these featurest o create a new feature-extraction model that returns
    # the values of the intermediate layer activations
    feature_extraction_model = keras.Model(
        inputs = vgg19.input,
        outputs=features_list,
    )

    image = np.random.random((1, 224, 224, 3)).astype('float32')

    # Pass image through the feature extraction model
    extracted_features = feature_extraction_model(image)
    
    # This comes in handy for tasks like neural style transfer

'''
Extend the API using custom layers

`tf.keras` includes a range of built-in layers, for example:

    - Convolutional layers : `Conv1D`, `Conv2D`, `Conv3D`, `Conv2DTranspose`
    - Pooling layers : `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `AveragePooling1D`
    - RNN layers : `GRU`, `LSTM`, `ConvLSTM2D`
    - `BatchNormalization`, `Dropout`, `Embedding`

To create your own layers:

    - Override the `call` method
    - Override the `built` method

'''
class Custom_Dense_Example9(layers.Layer):
    def __init__(self, units=32):
        super(Custom_Dense_Example9, self).__init__()
        self.units = units

    # Defines the weights w and b in y = wx + b
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Here we make use of our custom Dense layer
def example9():

    # Define input layer
    inputs = keras.Input((4,))

    # Use custom layer with 10 units
    outputs = Custom_Dense_Example9(10)(inputs)

    # Create model, print summary
    model = keras.Model(inputs, outputs)
    model.summary()

    # Graph the model for visualization
    keras.utils.plot_model(model, './model_images/custom_dense_model_9.png', show_shapes=True)


'''
For serialization support in your custom layer, define a `get_config`
method that returns the constructor arguments of the layer instance:
'''
class Custom_Dense_Example10(layers.Layer):
    def __init__(self, units=32):
        super(Custom_Dense_Example10, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        return {'units' : self.units}

def example10():
    
    inputs = keras.Input((4,))
    outputs = Custom_Dense_Example10(10)(inputs)

    model = keras.Model(inputs = inputs, outputs = outputs)

    config = model.get_config()

    new_model = keras.Model.from_config(config, custom_objects = {'Custom_Dense_Example10':Custom_Dense_Example10})

    new_model.summary()

    keras.utils.plot_model(new_model, './model_images/custom_dense_model_10.png', show_shapes=True)

'''
Mix and match API styles

It does not support dynamic architectures

'''

class CustomRNN(layers.Layer):
    def __init__(self, units, model):
        super(CustomRNN, self).__init__()

        # Consolidate missing vars -->
        self.layers = model.layers
        self._is_graph_network = model._is_graph_network
        self._network_nodes = model._network_nodes
        # <--/
        
        self.units = units
        self.projection_1 = layers.Dense(units=self.units, activation='tanh')
        self.projection_2 = layers.Dense(units=self.units, activation='tanh')

        # Our previously defined functinal model
        self.classifier = model

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))

        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        
        features = tf.stack(outputs, axis=1)
        print(features.shape)
        return self.classifier(features)

def customRNNexample():
    units = 32
    timesteps = 10
    input_dim = 5
    
    # Define a functional model
    inputs = keras.Input((None, units))
    x = layers.GlobalAveragePooling1D()(inputs)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    rnn_model = CustomRNN(units, model)
    _ = rnn_model(tf.zeros((1, timesteps, input_dim)))

    keras.utils.plot_model(rnn_model, './model_images/custom_rnn.png', show_shapes=True)

if __name__ == "__main__":
    main()