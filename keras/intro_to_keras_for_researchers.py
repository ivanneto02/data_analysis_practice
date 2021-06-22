import tensorflow as tf
from tensorflow import keras

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
    #example11()
    #example12()
    pass

def example1():

    '''Constant tensor'''
    x = tf.constant([[5, 2], [1, 3]])
    print(x)

    '''You can get its value as a NumPy array by calling `.numpy()'''
    print(x.numpy())

    '''Much like a Numpy array, it features the attributes `dtype` and `shape`'''
    print('dtype:', x.dtype)
    print('shape:', x.shape)

def example2():
    '''A common way to create constant tensors is via `tf.ones` and `tf.zeros` (just like `np.ones` and `np.zeros`)'''
    print(tf.ones(shape=(2, 1)))
    print(tf.zeros(shape=(2, 1)))

def example3():
    '''You can also create random constant tensors'''
    x = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
    print('\nrandom x:\n', x)
    x = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype='int32')
    print('\nrandom x uniform:\n', x)

def example4():
    '''
    Variables

    Variables are special tensors used to store mutable state (such as weights of a neural network).
    You can create a `Variable` using some initial value:
    '''
    initial_value = tf.random.normal(shape=(2, 2))

    a = tf.Variable(initial_value)
    print(a)

    '''
    You update the value of a `Variable` by using the methods `.assign(value)`, `.assign_add(increment)`, or `.assign_sub(decrement)`:
    '''
    new_value = tf.random.normal(shape=(2, 2))
    a.assign(new_value)

    for i in range(2):
        for j in range(2):
            assert a[i, j] == new_value[i, j]
    
def example5():
    '''
    Doing math in TensorFlow

    If you've used NumPy, doing math in TensorFlow will look very familiar.
    The main difference is that your TensorFlow code can run on GPU and  TPU.
    '''

    # Random a and b
    a = tf.random.normal(shape=(2, 2))
    b = tf.random.normal(shape=(2, 2))

    # Add
    c = a + b

    # Square
    d = tf.square(c)

    # Exponential
    e = tf.exp(d)

def example6():
    '''
    Gradients

    Here's another big difference with NumPy: you can automatically retrieve the gradient of any differentiable expression.

    Just open a `GradientTape`, start watching a tensor via `tape.watch()`,
    and compose a differentiable expression using this tensor as input:
    '''

    a = tf.random.normal(shape=(2, 2))
    b = tf.random.normal(shape=(2, 2))

    with tf.GradientTape() as tape:
        tape.watch(a) # Start recording the history of operations applied to `a`
        c = tf.sqrt(tf.square(a) + tf.square(b)) # Pythagoeran theorem

        # What's the gradient of `c` with respect to `a`?
        dc_da = tape.gradient(c, a)
        print(dc_da)

    # By default, variables are watched automatically, so you don't need to manually `watch` them:

    a = tf.Variable(a)  # Notice a is outside GradientTape() call

    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
        print(dc_da)

    # Notice that you can compute higher-order derivatives by nesting tapes:
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            c = tf.sqrt(tf.square(a) + tf.square(b))

            # First order gradient
            dc_da = inner_tape.gradient(c, a)

        # Second order gradient
        d2c_da2 = outer_tape.gradient(dc_da, a)
        print(d2c_da2)

def example7():
    '''
    Keras layers

    While TensorFlow is an infrastructure layer for differentiable programming,
    dealing with tensors, variables, and gradients,
    Keras is a user interface for deep learning, dealing with
    layers, models, optimizers, loss functions, metrics, and more.

    Keras serves as the high-level API for TensorFlow:
    Keras is what makes TensorFlow simple and productive.

    The `Layer` class is the fundamental abstraction in Keras.
    A `Layer` encapsulates a state (weights) and some computation
    (defined in the call method).

    A simple layer looks like this:
    '''

    class Linear(keras.layers.Layer):
        # y = w.x + b

        def __init__(self, units=32, input_dim=32):
            super(Linear, self).__init__()

            # w weight
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(
                initial_value = w_init(shape=(input_dim, units), dtype='float32'),
                trainable = True
            )

            # b weight
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(

                initial_value = b_init(shape=(units,), dtype='float32'),
                trainable = True,
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b
        
    # You would use a layer instance much like a Python function

    units = 4
    input_dim = 2

    # Instantiate layer
    linear_layer = Linear(units=units, input_dim=input_dim)

    # The layer can be treated as a function.
    # Here we call it on some data.
    y = linear_layer(tf.ones((2, 2)))
    assert y.shape == (2, 4)

    # The weight variables created in `__init__` are automatically tracked
    # under the `weights` property:
    assert linear_layer.weights == [linear_layer.w, linear_layer.b]

def example8():
    '''
    You have many built-in layers available, from `Dense` to `Conv2D` to `LSTM` to fancier
    ones like `Conv3DTranspose` or `ConvLSTM2D`. Be smart about reusing built-in functionality.
    '''

    '''
    Layer weight creation

    The `self.add_weight()` method gives you a shortcut for creating weight:
    '''

    class Linear(keras.layers.Layer):
        # y = w.x + b

        def __init__(self, units=32):
            super(Linear, self).__init__()
            self.units = units
        
        def build(self, input_shape):
            # weight w
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True,
            )

            # weight b
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='random_normal',
                trainable=True,
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    # Instantiate our lazy layer.
    linear_layer = Linear(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(tf.ones((2, 2)))

    '''
    Layer gradients

    You can automatically retrieve the gradients of the weights of a layer by
    calling it inside a `GradientTape`. Using these gradients, you can update the
    weights of the layer, either manually, or using an optimizer object. Of course,
    you can modify the gradients before using them, if you need to.
    '''

    # Prepare a dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 784).astype('float32') / 255.0, y_train)
    )

    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    # Instantiate our linear layer (defined above) with 10 units.
    linear_layer = Linear(10)

    # Instantiate a logistic loss function that expects integer targets.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Instantiate an optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    epochs = 20

    for epoch in range(0, epochs):

        print(' Epoch: {}'.format(epoch))

        # Iterate over the batches of the dataset
        for step, (x, y) in enumerate(dataset):

            # Open a GradientTape
            with tf.GradientTape() as tape:

                # Forward pass
                logits = linear_layer(x)

                # Loss value for this batch
                loss = loss_fn(y, logits)

            # Get gradients of weights
            gradients = tape.gradient(loss, linear_layer.trainable_weights)

            # Update the weights of our linear layer
            optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))

            # Logging
            if step % 100 == 0:
                print('Step:', step, "Loss:", float(loss))
        
def example9():
    '''
    Trainable and non-trainable weights

    Weights created by layers can either be trainable or non-trainable. They're
    exposed in `trainable_weights` and `non_trainable_weights` respectively.
    Here's a layer wih a non-trainable weight:
    '''

    class ComputeSum(keras.layers.Layer):
        '''Returns the sum of the inputs.'''

        def __init__(self, input_dim):
            super(ComputeSum, self).__init__()

            # Create a non-trainable weight
            self.total = tf.Variable(
                initial_value = tf.zeros((input_dim,)),
                trainable = False,
            )

        # Computes the addition of the inputs and returns the total
        def call(self, inputs):
            self.total.assign_add(tf.reduce_sum(inputs, axis = 0))
            return self.total

    # Example of instance
    my_sum = ComputeSum(2)
    
    x = tf.ones((2, 2))

    y = my_sum(x)
    print(y.numpy())

    y = my_sum(x)
    print(y.numpy())

    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []

def example10():
    '''
    Layers that can own layers

    Layers can be recursively nested to create igger computation blocks.
    Each layer will track the weights of it sublayers
    (both trainable and non-trainable)
    '''

    # Let's reuse the Linear class
    # with a `build` method that we defined above.
    class Linear(keras.layers.Layer):
        # y = w.x + b

        def __init__(self, units=32):
            super(Linear, self).__init__()
            self.units = units
        
        def build(self, input_shape):
            # weight w
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True,
            )

            # weight b
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='random_normal',
                trainable=True,
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    class MLP(keras.layers.Layer):
        '''Simple stack of Linear layers.'''

        def __init__(self):
            super(MLP, self).__init__()

            # Data members
            self.linear_1 = Linear(32)
            self.linear_2 = Linear(32)
            self.linear_3 = Linear(10)

        def call(self, inputs):
            
            # Layers
            x_in = inputs
            x = self.linear_1(x_in)
            x = tf.nn.relu(x)
            x = self.linear_2(x)
            x = tf.nn.relu(x)
            x_out = self.linear_3(x)
            return x_out

    # MLP instance
    mlp = MLP()

    # The first call to the `mlp` object will create the weights
    y = mlp(tf.ones(shape=(3, 64)))

    # Weights are recursively tracked.
    assert len(mlp.weights) == 6

    # Note that our manually-created MLP above is equivalent to the following
    # built-in option:

    mlp = keras.Sequential(

        [
            keras.layers.Dense(32, activation=tf.nn.relu),
            keras.layers.Dense(32, activation=tf.nn.relu),
            keras.layers.Dense(10),
        ],

    )

def example11():
    '''
    Tracking losses created by layers

    Layers can create losses during the forward pass via the `add_loss()` method.
    This is especially useful for regularization losses.
    The losses created by subplayers are recursively tracked by the parent layers.

    Here's a layer that creates an activity regularization loss:
    '''
    class Linear(keras.layers.Layer):
        # y = w.x + b

        def __init__(self, units=32):
            super(Linear, self).__init__()
            self.units = units
        
        def build(self, input_shape):
            # weight w
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True,
            )

            # weight b
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='random_normal',
                trainable=True,
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    class ActivityRegularization(keras.layers.Layer):
        '''Layer that creates an activity sparsity regularization loss.'''

        def __init__(self, rate=1e-2):
            super(ActivityRegularization, self).__init__()

            # Data members
            self.rate = rate

        def call(self, inputs):
            # We use `add_loss` to create a regularization loss
            # that depends on the inputs
            self.add_loss(self.rate * tf.reduce_sum(inputs))
            return inputs

    '''
    Any model incorporating this layer will track this regularization loss:
    '''

    # Let's use the loss layer in a MLP block.

    class SparseMLP(keras.layers.Layer):
        '''Stack of Linear layers with a sparsity regularization loss.'''

        def __init__(self):
            super(SparseMLP, self).__init__()

            # Data members
            self.linear_1 = Linear(32)
            self.regularization = ActivityRegularization(1e-2)
            self.linear_3 = Linear(10)
        
        def call(self, inputs):
            
            # Layer calls
            x_in = inputs
            x = self.linear_1(x_in)
            x = tf.nn.relu(x)
            x = self.regularization(x)
            x_out = self.linear_3(x)
            return x_out

    mlp = SparseMLP()

    y = mlp(tf.ones((10, 10)))

    print(mlp.losses) # List the containing one float32 scalar

    '''
    These losses are cleared by the top-level layer at the start of each forward
    pass -- they don't accumulate. `layer.losses` always contains only the losses
    created during the last forward pass. You would typically use these losses by 
    summing them before computing your gradients when writing a training loop.
    '''

    # Losses correspond to the last forward pass.
    mlp = SparseMLP()

    mlp(tf.ones((10, 10)))
    assert len(mlp.losses) == 1

    mlp(tf.ones((10, 10)))
    assert len(mlp.losses) == 1 # There is no accumulation

    # Demonstration in a training loop

    # Pepare dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 784).astype('float32') / 255.0, y_train)
    )

    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    # A new MLP
    mlp = SparseMLP()

    # Loss and optimizer.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    epochs = 1

    for epoch in range(0, epochs):

        print(' Epoch: {}'.format(epoch))

        for step, (x, y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                
                # Forward pass.
                logits = mlp(x)

                # External loss value for this batch
                loss = loss_fn(y, logits)

                # Add the losses created during the forward pass.
                loss += sum(mlp.losses)

                # Get gradients of weights wrt the loss.
                gradients = tape.gradient(loss, mlp.trainable_weights)

            # Update the weights of our linear layer
            optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))

            # Logging

            if step % 100 == 0:
                print('Step:', step, 'Loss:', float(loss))

    '''
    Keeping track of training metrics

    Keras offers a broad range of build-in metrics, like `tf.keras.metrics.AUC`
    of `tf.keras.metrics.PrecisionAtRecall`. It's also easy to create your own
    metric in a few lines of code.

    To use a metric in a custom training loop, you would:
        
        - Instantiate the metric object, e.g. `metric = tf.keras.metrics.AUC()`
        - Call its `metric.update_state(target, predictions)` methods for each batch of data
        - Query its result via `metric.result()`
        - Reset the metric's state at the end of an epoch or at the start of an evaluation via `metric.reset_states()`

    Here's a simple example:
    '''        

    # Instantiate a metric object
    accuracy = tf.keras.metrics.SparseCategoricalCrossentropy()

    # Prepare our layer, loss, and optimizer.
    model = keras.Sequential(

        [
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10),
        ],

    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    epochs = 2

    for epoch in range(0, epochs):
        # Iterate over the batches of a dataset
        
        print(' Epoch: {}'.format(epoch))

        for step, (x, y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(x)

                # Compute the loss value for this batch.
                loss_value = loss_fn(y, logits)
            
            # Update the state of the `accuracy` metric.
            accuracy.update_state(y, logits)

            # Update the weights of the model to minimize the loss value.
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            # Logging the current accuracy value so far.
            if step % 100 == 0:
                print('Step:', step)
                print('Total running accuracy so farL %.3f' % accuracy.result())

        # Reset the metric's state at the end of an epoch
        accuracy.reset_states()

    '''
    In addition to this, similarly to the `self.add_loss()` method, you have access
    to an `self.add_metric()` method on layers. It tracks the average of whatever
    quantity you pass to it. You can reset the value of these metrics by
    calling `layer.reset_metrics()` on any layer or model.
    '''

    '''
    Compiled Functions

    Running eagerly is great for debugging, but you will get better performance by
    compiling your computation into static graphs. Static graphs are a researcher's
    best friends. You can compile any function by wrapping it in a `ft.function`
    decorator.
    '''

    # Prepare our layer, loss, and optimizer.
    model = keras.Sequential(
        [
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10),
        ],
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Create a training step function.

    @tf.function # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = loss_fn(y, logits)
            gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return loss
    
    # Prepare a dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(60000, 784).astype('float32') / 255.0, y_train)
    )

    dataset = dataset.shuffle(buffer_size=1024).batch(64)

    epochs = 10

    for epoch in range(0, epochs):

        print(' Epoch: {}'.format(epoch))

        for step, (x, y) in enumerate(dataset):
            loss = train_on_batch(x, y)
            if step % 100 == 0:
                print('Step:', step, 'Loss:', float(loss))

def example12():
    '''
    Training mode & inference mode

    Some layers, in particular the `BatchNormalization` layer and the `Dropout`
    layer, have different behaviors during training and inference. For such layers,
    it is standard practice to expose a `training` (boolean) argument in the `call`
    method.

    By exposing this argument in `call`, you enable the built-in training and
    evaluation loops (e.g. fit) to correctly use the layer in training and 
    inference modes.
    '''

    class Linear(keras.layers.Layer):
        # y = w.x + b

        def __init__(self, units=32):
            super(Linear, self).__init__()
            self.units = units
        
        def build(self, input_shape):
            # weight w
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True,
            )

            # weight b
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='random_normal',
                trainable=True,
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    class Dropout(keras.layers.Layer):
        def __init__(self, rate):
            super(Dropout, self).__init__()
            
            # Data members
            self.rate = rate
        
        def call(self, inputs, training=None):
            if training:
                return tf.nn.dropout(inputs, rate=self.rate)
            
            return inputs

    class MLPWithDropout(keras.layers.Layer):
        def __init__(self):
            super(MLPWithDropout, self).__init__()

            # Data members
            self.linear_1 = Linear(32)
            self.dropout = Dropout(0.5)
            self.linear_3 = Linear(10)
        
        def call(self, inputs, training=None):
            
            # Layers
            x_in = inputs
            x = self.linear_1(x_in)
            x = tf.nn.relu(x)
            x = self.dropout(x, training=training)
            x_out = self.linear_3(x)
            return x_out

    mlp = MLPWithDropout()
    
    y_train = mlp(tf.ones((2, 2)), training=True)
    y_test = mlp(tf.ones((2, 2)), training=False)

    '''
    The Functional API for model-building

    To build deep learning models, you don't have to use object-oriented programming all the
    time. All layers we've seen so far can also be composed functi
    '''

    # We use an `Input` object to describe the shape and dtype of the inputs.
    # This is the deep learning equivalent of declaring a type.
    # The shape argument is per-sample; it does not include the batch size.
    # The functional API focused on defining per-sample transformations.
    # The model we create will automatically batch the per-sample transformations,
    # so that it can be called on batches of data.
    inputs = tf.keras.Input(shape=(16,), dtype='float32')

    # We call layers on these "type" objects
    # and they return updated types (new shapes / dtypes)
    x_in = inputs
    x = Linear(32)(x_in)    # We are reusing the Linear layer we defined earlier.
    x = Dropout(0.5)(x)     # We are reusing the Dropout layer we defined earlier.
    outputs = Linear(10)(x)

    # A functional `Model` can be defined by specifying inputs and outputs.
    # A model is itself a layer like any other.
    model = tf.keras.Model(inputs, outputs)

    # A functional model already has weights, before being called on any data.
    # That's because we defined its input shape in advance (in `Input`).
    assert len(model.weights) == 4

    # Let's call our model on some data, for fun.
    y = model(tf.ones((2, 16)))
    assert y.shape == (2, 10)

    # You can pass a `training` argument in `__call__`
    # (it will get passed down to the Dropout layer).
    y = model(tf.ones((2, 16)), training=True)

if __name__ == "__main__":
    main()