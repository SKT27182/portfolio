import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class AutoEncoder:
    """
    AutoEncoder class

    Parameters
    ----------
    input_shape : tuple
        Input shape of the data

    Attributes
    ----------
    input_shape : tuple

    Methods
    -------
    dense_encoder(inputs, neurons, activation="relu")
        Dense encoder model

    bottlenck_dense(inputs, activation="relu")
        Bottlenck layer

    dense_decoder(inputs, neurons, activation="relu")
        Dense decoder model

    dense_autoencoder(neurons, activation="relu")
        Dense autoencoder model

    conv_compress(inputs, filter, kernel_size, strides, activation="relu", name="")
        Convolutional compress layer

    conv_encoder(inputs, filters, kernel_size=3, strides=1, activation="relu")
        Convolutional encoder model

    bottlenck_conv(inputs, activation="relu")
        Bottlenck layer

    conv_expand(inputs, filter, kernel_size, strides, activation="relu", name="")
        Convolutional expand layer

    conv_decoder(inputs, filters, kernel_size=3, strides=1, activation="relu")
        Convolutional decoder model

    conv_autoencoder(filters, kernel_size=3, strides=1, activation="relu")
        Convolutional autoencoder model

    build(neurons, filters, kernel_size=3, strides=1, activation="relu")
        Build the model

    compile(optimizer, loss, metrics)
        Compile the model

    fit(x, y, epochs, batch_size, validation_split)
        Fit the model

    evaluate(x, y, batch_size)
        Evaluate the model
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def dense_encoder(self, inputs, neurons, activation):
        x = inputs

        for i, neuron in enumerate(neurons):
            x = tf.keras.layers.Dense(
                neuron, activation=activation, name=f"encoder_{i}"
            )(x)

        encoded_x = self.bottlenck_dense(x)[0]

        encoder_model = tf.keras.Model(inputs, encoded_x)

        return encoded_x, encoder_model

    def bottlenck_dense(self, inputs):
        x = inputs

        x = tf.keras.layers.Dense(
            self.encoding_shape, activation="relu", name="bottlenck"
        )(x)

        encoded_model = tf.keras.Model(inputs, x)

        return x, encoded_model

    def dense_decoder(self, inputs, neurons, activation):
        x = inputs

        for i, neuron in enumerate(reversed(neurons)):
            x = tf.keras.layers.Dense(
                neuron, activation=activation, name=f"decoder_{i}"
            )(x)

        decoded_x = tf.keras.layers.Dense(
            self.input_shape, activation="relu", name="decode_output"
        )(x)

        decoder_model = tf.keras.Model(inputs, decoded_x)

        return decoded_x, decoder_model

    def dense_autoencoder(self, neurons, activation="relu"):
        inputs = tf.keras.Input(shape=self.input_shape)

        encoded_x, encoder_model = self.dense_encoder(
            inputs, neurons, activation=activation
        )

        decoded_x, decoder_model = self.dense_decoder(
            encoded_x, neurons, activation=activation
        )

        autoencoder = tf.keras.Model(inputs, decoded_x)

        return encoder_model, decoder_model, autoencoder

    def conv_compress(
        self, inputs, filter, activation, name
    ):
        x = inputs

        x = tf.keras.layers.Conv2D(
            filter,
            kernel_size=3,
            activation=activation,
            padding="same",
            name=f"{name}_Conv1",
        )(x)

        x = tf.keras.layers.Conv2D(
            filter,
            kernel_size=3,
            activation=activation,
            padding="same",
            name=f"{name}_Conv2",
        )(x)

        # to compress the image
        compressed_x = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, name=f"{name}_MaxPool"
        )(x)

        encoder_model = tf.keras.Model(inputs, compressed_x)

        return compressed_x, encoder_model

    def conv_encoder(
        self, inputs, filters, activation, encoding_depth
    ):
        x = inputs

        for i, filter in enumerate(filters):
            x, encoder_model = self.conv_compress(
                x,
                filter,
                activation=activation,
                name=f"encoder_{i}",
            )

        encoded_x = self.bottlenck_conv(x, encoding_depth=encoding_depth)[0]

        encoded_model = tf.keras.Model(inputs, encoded_x)

        return encoded_x, encoded_model

    def bottlenck_conv(self, inputs, encoding_depth):
        x = inputs

        x = tf.keras.layers.Conv2D(
            filters=encoding_depth, kernel_size=3, activation="relu", padding="same", name="bottlenck"
        )(x)

        encoded_model = tf.keras.Model(inputs, x)

        return x, encoded_model

    def conv_expand(
        self, inputs, filter,  activation, name
    ):
        x = inputs

        x = tf.keras.layers.Conv2DTranspose(
            filter,
            kernel_size=3,
            activation=activation,
            padding="same",
            name=f"{name}_ConvT1",
        )(x)

        # upsample the image
        x = tf.keras.layers.UpSampling2D(size=2, name=f"{name}_UpSample")(x)

        expanded_x = tf.keras.layers.Conv2DTranspose(
            filter,
            kernel_size=3,
            activation=activation,
            padding="same",
            name=f"{name}_ConvT2",
        )(x)

        decoder_model = tf.keras.Model(inputs, expanded_x)

        return expanded_x, decoder_model

    def conv_decoder(
        self, inputs, filters, activation
    ):
        x = inputs

        for i, filter in enumerate(reversed(filters)):
            x, decoder_model = self.conv_expand(
                x,
                filter,
                activation=activation,
                name=f"decoder_{i}",
            )

        decoded_x = tf.keras.layers.Conv2DTranspose(
            self.input_shape[-1], kernel_size=3, activation="relu",padding="same" , name="decode_output"
        )(x)

        decoder_model = tf.keras.Model(inputs, decoded_x)

        return decoded_x, decoder_model

    def conv_autoencoder(self, filters, activation="relu", encoding_depth=128):
        inputs = tf.keras.Input(shape=self.input_shape)

        encoded_x, encoder_model = self.conv_encoder(
            inputs, filters, activation, encoding_depth
        )

        decoded_x, decoder_model = self.conv_decoder(
            encoded_x, filters, activation=activation
        )

        autoencoder = tf.keras.Model(inputs, decoded_x)

        return encoder_model, decoder_model, autoencoder

    def build(self, neurons=None, encoding_shape=None, filters=None, kernel_size=None, activation="relu", encoding_depth=128):
        """
        Builds the autoencoder model
        
        Parameters
        ----------
        
        neurons : list
            List of neurons in each layer of the encoder and decoder
            
        encoding_shape : tuple
            Shape of the encoding layer
        
        filters : list
            List of filters in each layer of the encoder and decoder
        
        kernel_size : int
            Kernel size of the convolutional layers
        
        activation : str
            Activation function to use in the layers
            
        """
        if neurons is not None:
            self.encoding_shape = encoding_shape
            self.encoder, self.decoder, self.autoencoder = self.dense_autoencoder(
                neurons, activation=activation
            )
            return
        elif filters is not None:
            self.encoder, self.decoder, self.autoencoder = self.conv_autoencoder(
                filters, activation=activation, encoding_depth=encoding_depth
            )
            return
        else:
            raise ValueError("Invalid parameters")   

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        """
        Compiles the autoencoder model
        (It is necessary to first build the model before compiling it)
        
        Parameters
        ----------
        
        optimizer : str
            Optimizer to use for training
            
        loss : str
            Loss function to use for training
            
        metrics : list
            List of metrics to use for training
            
        **kwargs : dict
            Keyword arguments to pass to the compile method
            
        """

        if self.autoencoder is None:
            raise ValueError("Model not built yet")
        self.autoencoder.compile(optimizer, loss, metrics, **kwargs)
     
    def summary(self):
        """

        Prints the summary of the autoencoder model
        (It is necessary to first build and compile the model before printing the summary)

        """

        if self.autoencoder is None:
            raise ValueError("Model not built yet")
        elif self.autoencoder.optimizer is None:
            raise ValueError("Model not compiled yet")
        self.autoencoder.summary()

    def fit(self, x, y, epochs=1, batch_size=32, validation_data=None, **kwargs):
        """

        Trains the autoencoder model

        Parameters
        ----------

        x : array
            Input data

        y : array
            Target data

        epochs : int
            Number of epochs to train the model

        batch_size : int
            Batch size to use for training

        validation_data : tuple
            Validation data to use for training

        **kwargs : dict
            Keyword arguments to pass to the fit method

        """

        if self.autoencoder is None:
            raise ValueError("Model not built yet")
        elif self.autoencoder.optimizer is None:
            raise ValueError("Model not compiled yet")

        self.autoencoder.fit(
            x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, **kwargs
        )

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)
    
    def evaluate(self, inputs, input_shape, encoded_shape, plot=False, num=10):
        
        encoded_img = self.encode(inputs)
        decoded_img = self.decode(encoded_img)

        if (encoded_img.shape[-1] > 3) and len(encoded_img.shape) == 4:
            encoded_img = tf.math.reduce_mean(encoded_img, axis=-1, keepdims=True)

        if plot:
            
            plt.figure(figsize=(20, 4))
            for i in range(10):
                # display original
                ax = plt.subplot(3, 10, i + 1)
                plt.imshow(inputs[i].numpy().reshape(input_shape))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display encoded
                ax = plt.subplot(3, 10, i + 1 + 10)
                plt.imshow(encoded_img[i].numpy().reshape(encoded_shape))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                

                # display reconstruction
                ax = plt.subplot(3, 10, i + 1 + 20)
                plt.imshow(decoded_img[i].numpy().reshape(input_shape))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)