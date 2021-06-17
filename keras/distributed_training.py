import tensorflow as tf
from tensorflow import keras
import os

def main():

    #example1()
    example2()

def get_compiled_model():

    # Make a simple 2-layer densely-connected neural network

    inputs = keras.Input(shape=(784,))

    x = keras.layers.Dense(256, activation='relu')(inputs)
    x = keras.layers.Dense(256, activation='relu')(x)

    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

def get_dataset():

    batch_size = 32
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]

    return (

        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),

    )

def example1():

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open strategy scope
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = get_compiled_model()

    # Train the model on all available devices.
    train_dataset, val_dataset, test_dataset = get_dataset()
    model.fit(train_dataset, epochs=2, validation_data=val_dataset)

    # Test the model on all available devices.
    model.evaluate(test_dataset)

def make_or_restore_model(checkpoint_dir):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key = os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    
    print("Creating new model")
    return get_compiled_model()

def run_training(epochs, train_dataset, val_dataset, checkpoint_dir):
    # Create a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model(checkpoint_dir)

    callbacks = [

        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name
        keras.callbacks.ModelCheckpoint(

            filepath = checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        )
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=2,
    )

def example2():
    checkpoint_dir = './ckpt'

    # Parameters
    epochs = 10

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    train_dataset, val_dataset, test_dataset = get_dataset()

    run_training(epochs, train_dataset, val_dataset, checkpoint_dir)

if __name__ == "__main__":
    main()