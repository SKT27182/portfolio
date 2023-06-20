import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm


class GAN(object):
    def __init__(self, latent_dim=16):
        """Initialize the GAN model

        Parameters:
        -----------

        latent_dim (int): default=16
            The dimension of the latent space

        Returns:
        --------

        None

        """
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan_model = None

    def __repr__(self):
        return f"GAN(latent_dim={self.latent_dim})"

    def __str__(self):
        return f"GAN(latent_dim={self.latent_dim})"

    def dense_generator(self, neurons, activation, name):
        """
        Create a dense generator model

        Parameters:
        -----------

        inputs (tf.Tensor):
            The input tensor

        neurons (list):
            The number of neurons in each layer

        activation (list):
            The activation function for each layer

        name (str): default="dense_generator"
            The name of the model

        Returns:
        --------

        tf.Tensor:
            The output tensor

        """

        inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name=f"input")

        x = inputs

        for i, n in enumerate(neurons):
            x = tf.keras.layers.Dense(n)(x)
            x = tf.keras.layers.Activation(activation)(x)

        x = tf.keras.layers.Dense(tf.reduce_prod(self.output_shape).numpy())(x)

        x = tf.keras.layers.Activation("tanh")(x)

        x = tf.keras.layers.Reshape(self.output_shape, name=f"output")(x)

        self.generator = tf.keras.models.Model(inputs=inputs, outputs=x, name=name)

    def dense_discriminator(self, neurons, name):
        """
        Create a dense discriminator model

        Parameters:
        -----------

        inputs (tf.Tensor):
            The input tensor

        neurons (list):
            The number of neurons in each layer

        activation (list):
            The activation function for each layer

        name (str): default="dense_discriminator"
            The name of the model

        Returns:
        --------

        tf.Tensor:
            The output tensor

        """

        inputs = tf.keras.layers.Input(shape=self.output_shape, name=f"input")

        x = inputs

        x = tf.keras.layers.Flatten()(x)

        for _, n in enumerate(neurons):
            x = tf.keras.layers.Dense(n)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Dense(1, name=f"output")(x)

        x = tf.keras.layers.Activation("sigmoid")(x)

        self.discriminator = tf.keras.models.Model(inputs=inputs, outputs=x, name=name)

    def conv_gen_block(
        self, inputs, filters, kernel_size, strides, activation, name, padding="same"
    ):
        """
        Create a conv block

        Parameters:
        -----------

        inputs (tf.Tensor):
            The input tensor

        filters (int):
            The number of filters in each layer

        kernel_size:
            The kernel size

        strides :
            The strides

        activation :
            The activation function

        name (str): default="conv_block"
            The name of the model

        Returns:
        --------

        tf.Tensor:
            The output tensor

        """

        x = inputs

        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            use_bias=False,
            padding=padding,
            name=f"{name}_conv",
        )(inputs)

        x = tf.keras.layers.BatchNormalization(name=f"{name}_batchnorm")(x)

        x = tf.keras.layers.Activation(activation, name=f"{name}_activation")(x)

        return x

    def conv_generator(self, filters, kernel_size, activation, name):
        """
        Create a conv generator model

        Parameters:
        -----------

        inputs (tf.Tensor):
            The input tensor

        filters (list):
            The number of filters in each layer

        kernel_size:
            The kernel size

        strides :
            The strides

        activation :
            The activation function

        name (str): default="conv_generator"
            The name of the model

        Returns:
        --------

        tf.Tensor:
            The output tensor

        """

        inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name=f"input")

        x = inputs

        x = tf.keras.layers.Reshape((1, 1, self.latent_dim), name=f"reshape")(x)

        x = self.conv_gen_block(
            x, filters[0], kernel_size, 1, activation, f"conv_block_0", padding="valid"
        )

        for i, f in enumerate(filters[1:]):
            x = self.conv_gen_block(
                x, f, kernel_size, 2, activation, f"conv_block_{i+1}"
            )

        x = tf.keras.layers.Conv2DTranspose(
            self.output_shape[-1],
            kernel_size,
            strides=2,
            use_bias=False,
            padding="same",
            name=f"output_conv",
        )(x)

        # resize to output shape
        x = tf.image.resize(x, self.output_shape[:-1])

        x = tf.keras.layers.Activation("tanh", name=f"output")(x)

        self.generator = tf.keras.models.Model(inputs=inputs, outputs=x, name=name)

    def conv_descr_block(
        self, inputs, filters, kernel_size, strides, name, padding="same"
    ):
        """
        Create a conv block

        Parameters:
        -----------

        inputs (tf.Tensor):
            The input tensor

        filters (int):
            The number of filters in each layer

        kernel_size:
            The kernel size

        strides :
            The strides

        name (str): default="conv_block"
            The name of the model

        Returns:
        --------

        tf.Tensor:
            The output tensor

        """

        x = inputs

        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            use_bias=False,
            padding=padding,
            name=f"{name}_conv",
        )(inputs)

        x = tf.keras.layers.BatchNormalization(name=f"{name}_batchnorm")(x)

        x = tf.keras.layers.LeakyReLU(0.2, name=f"{name}_activation")(x)

        return x

    def conv_discriminator(self, filters, kernel_size, name):
        """
        Create a conv discriminator model

        Parameters:
        -----------

        inputs (tf.Tensor):
            The input tensor

        filters (list):
            The number of filters in each layer

        kernel_size:
            The kernel size

        strides :
            The strides

        activation :
            The activation function

        name (str): default="conv_discriminator"
            The name of the model

        Returns:
        --------

        tf.Tensor:
            The output tensor

        """

        inputs = tf.keras.layers.Input(shape=self.output_shape, name=f"input")

        x = inputs

        x = self.conv_descr_block(x, filters[0], kernel_size, 2, f"conv_block_0")

        for i, f in enumerate(filters[1:]):
            x = self.conv_descr_block(x, f, kernel_size, 2, f"conv_block_{i+1}")

        x = tf.keras.layers.Conv2D(
            1,
            kernel_size,
            strides=1,
            use_bias=False,
            padding="valid",
            name=f"{name}_output_conv",
        )(x)

        x = tf.keras.layers.Activation("sigmoid", name=f"output")(x)

        x = tf.keras.layers.Reshape((1,))(x)

        self.discriminator = tf.keras.models.Model(inputs=inputs, outputs=x, name=name)

    def build(
        self,
        output_shape=None,
        neurons=None,
        filters=None,
        kernel_size=4,
        activation="selu",
    ):
        """
        Create a GAN model

        Parameters:
        -----------

        output_shape (tf.Tensor):
            The shape of the output tensor

        neurons (list):
            The number of neurons in each layer

        activation (list):
            The activation function for each layer

        name (str): default="GAN"
            The name of the model

        Returns:
        --------

        None

        """

        self.output_shape = output_shape

        if neurons is not None:
            self.dense_generator(neurons, activation, name="generator")

            self.dense_discriminator(neurons[::-1], name="discriminator")

        else:
            self.conv_generator(filters, kernel_size, activation, name="generator")

            self.conv_discriminator(filters[::-1], kernel_size, name="discriminator")

    def compile(self, optimizer="adam", loss="binary_crossentropy"):
        """
        Compile the GAN model

        Parameters:
        -----------

        optimizer (str): default="adam"
            The optimizer to use

        loss (str): default="binary_crossentropy"
            The loss function to use

        Returns:
        --------

        None

        """

        self.gan_model = tf.keras.models.Sequential(
            [self.generator, self.discriminator]
        )

        self.discriminator.compile(optimizer=optimizer, loss=loss)
        self.discriminator.trainable = False

        self.gan_model.compile(optimizer=optimizer, loss=loss)

    def generate(self, noise):
        """
        Generate images

        Parameters:
        -----------

        noise (tf.Tensor):
            The noise to use

        Returns:
        --------

        tf.Tensor:
            The generated images

        """

        return self.generator(noise)

    def discriminate(self, images):
        """
        Discriminate images

        Parameters:
        -----------

        images (tf.Tensor):
            The images to discriminate

        Returns:
        --------

        tf.Tensor: , 0 for fake images, 1 for real images
            The discriminator output

        """

        return self.discriminator(images)

    def summary(self, **kwargs):
        self.generator.summary(**kwargs)
        print()
        self.discriminator.summary(**kwargs)

    def fit(
        self, x_train, epochs=10, show_image=False, show_interval=10, n_cols=4
    ):

        random_noise_for_generation = tf.random.normal(shape=(n_cols**2, self.latent_dim))

        # use tqdm to show the progress
        for epoch in range(epochs):
            step = 0

            for real_images in tqdm(x_train, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_size = real_images.shape[0]

                # Phase 1: Train the discriminator

                noise = tf.random.normal(shape=(batch_size, self.latent_dim))

                fake_images = self.generator(noise)

                mixed_images = tf.concat([fake_images, real_images], axis=0)

                # 0 for fake images, 1 for real images
                discriminator_labels = tf.constant(
                    [[0.0]] * batch_size + [[1.0]] * batch_size
                )

                self.discriminator.trainable = True

                self.discriminator.train_on_batch(mixed_images, discriminator_labels)

                # Phase 2: Train the generator

                noise = tf.random.normal(shape=(batch_size, self.latent_dim))

                # labels all 1 because we want the generator to fool the discriminator
                generator_labels = tf.constant([[1.0]] * batch_size)

                self.discriminator.trainable = False

                self.gan_model.train_on_batch(noise, generator_labels)

                if step % show_interval == 0 and show_image:

                    fake_images = self.generator(random_noise_for_generation)

                    self.show_images(fake_images, n_cols=n_cols)
                    plt.show()

                step += 1

    def show_images(self, images, n_cols=4):

        """visualizes fake images"""
        display.clear_output(wait=True)
        n_cols = n_cols or len(images)
        n_rows = (len(images) - 1) // n_cols + 1

        if images.shape[-1] == 1:
            cmap = "binary"
        else:
            cmap = None

        plt.figure(figsize=(n_cols, n_rows))

        for index, image in enumerate(images):
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(image, cmap=cmap)
            plt.axis("off")
