from math import e
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

def main():

    #first_highest()
    #second_highest()
    #third_highest()
    #fourth_highest()
    GAN_example()

## NOTE: Custom_Model classes are organized from highest level to lower level. Each of these classes corresponds to a function that calls the methods
## needed to train the models

## Making our own Custom_Model class where we derive from keras.Model.
## Then, we overrite train_step to customize our own fit.

class First_Highest_Custom_Model(keras.Model):
    def train_step(self, data):
        # Unpacks the data. Its structure depends on your model and on what you pass to `fit()`.

        # Data is (x, y)
        x, y = data[0], data[1]

        with tf.GradientTape() as tape:

            # Forward pass
            y_pred = self(x, training=True)

            # Compute the loss value
            # (The loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (include the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dictionary mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

# Highest level of abstraction
def first_highest():

    # Construct and compile an instance of CustomModel
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = First_Highest_Custom_Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Parameters
    epochs = 1000

    # Data
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))

    # Use `fit` as usual
    model.fit(x, y, epochs=epochs)

## Without passing a loss function to compile(), and instead doing everything manually in `train_step`.
## Metrics are also done in the same way.
## `compile` only configures the optimizer, which is Adam in this case.

class Second_Highest_Custom_Model(keras.Model):

    loss_tracker = keras.metrics.Mean(name="loss")
    MAE_metric = keras.metrics.MeanAbsoluteError(name="mae")

    def train_step(self, data):

        # Same step as before, set x and y as the data (i.e. data = (x, y))
        x, y = data[0], data[1]

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Compute our own MSR loss
            loss = keras.losses.mean_squared_error(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.MAE_metric.update_state(y, y_pred)

        # Return dictionary of the result for the loss and MAE
        return {"loss" : self.loss_tracker.result(), "mae": self.MAE_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.MAE_metric]

def second_highest():

    # Construct an instance of Second_highest_Custom_Model
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = Second_Highest_Custom_Model(inputs, outputs)

    # We don't pass a loss or metrics here.
    model.compile(optimizer="adam")

    # Parameters
    epochs = 50

    # Data
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))

    # Just use `fit` as usual -- yo ucan use callbacks, etc.
    model.fit(x, y, epochs=epochs)

## In this example, we are able to use `sample_weight` and `class_weight`
## We unpack sample_weight from data argument, and pass it to compiled_loss and compled_metrics

class Third_Highest_Custom_Model(keras.Model):
    def train_step(self, data):
        
        # Unpack the data. Its structure depends on your model and what you pass to `fit()``
        if len(data) == 3:
            x, y, sample_weight = data[0], data[1], data[2]
        else:
            sample_weight = None
            x, y = data[0], data[1]

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Compute the loss value
            # The loss function is configured in `compile()`
            loss = self.compiled_loss(
                y, y_pred,
                sample_weight = sample_weight,
                regularization_losses = self.losses
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {metric.name: metric.result() for metric in self.metrics}

def third_highest():
    
    # Construct and compile an instance of Third_Highest_Custom_Model
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = Third_Highest_Custom_Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Define data and sample weight
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    sample_weight = np.random.random((1000, 1))

    # Parameters
    epochs = 50

    # Fit the model
    model.fit(x, y, sample_weight=sample_weight, epochs=epochs)

## If you want to do the same for calls to `model.evaluate()`
## We override `test_step` in exactly the same way.

class Fourth_Highest_Custom_Model(keras.Model):
    def test_step(self, data):

        # Unpacks the data
        x, y = data[0], data[1]

        # Compute predictions
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {metric.name: metric.result() for metric in self.metrics}

def fourth_highest():
    
    # Construct an instance of Fourth_Highest_Custom_Model
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = Fourth_Highest_Custom_Model(inputs, outputs)
    model.compile(loss="mse", metrics=["mae"])

    # Evaluate with our custom test_step
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))

    model.evaluate(x, y)

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        
        gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(

            zip(gradients, self.discriminator.trainable_weights)

        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should not update the weights) of the discriminator!

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        gradients = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

def GAN_example():

    discriminator = keras.Sequential(

        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, (3,3), strides=(2,2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.GlobalMaxPooling2D(),
            layers.Dense(1),
        ],

        name = "discriminator",
    )

    # Create the generator
    latent_dim = 128

    generator = keras.Sequential(

        [
            keras.Input(shape=(latent_dim,)),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            layers.Dense(7 * 7 * 128),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((7, 7, 128)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],

        name = "generator",
    )

    # Parameters
    batch_size = 64
    epochs = 20
    learning_rate = 0.0003

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)

    gan.compile(
        d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True),
    )

    # To limit the execution time, we only train on 100 batches. You can train on the entire dataset. You will need
    # about 20 epochs to get nice results from this training.
    gan.fit(dataset.take(100), epochs=epochs)

if __name__ == "__main__":
    main()