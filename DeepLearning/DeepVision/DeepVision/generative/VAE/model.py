import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as display_fn
from PIL import Image
import imageio


class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for sampling from a probability distribution

    Parameters
    ----------
    args : tuple
        Arguments for the layer

    Methods
    -------
    call(inputs)
        Call the layer
    """

    def call(self, inputs):
        """
        Call the layer

        Parameters
        ----------
        inputs : tuple
            Inputs for the layer

            (mu, sigma) shape : (batch, dim)

        Returns
        -------
        z : tensor
            Sampled tensor
        """
        mu, sigma = inputs

        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return mu + tf.exp(0.5 * sigma) * epsilon


class VAE:
    def __init__(self, input_shape):
        """
        Initialize the VAE

        Parameters
        ----------
        input_shape : tuple
            Shape of the input

        """
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.vae = None

    def __repr__(self) -> str:
        return f"VAE(input_shape={self.input_shape})"
    

    def compress_block(self, inputs, filter, kernel_size, activation, name):
        x = inputs

        x = tf.keras.layers.Conv2D(
            filter,
            kernel_size,
            activation=activation,
            padding="same",
            name=name + "_conv1",
        )(x)

        x = tf.keras.layers.BatchNormalization(name=name + "_bn1")(x)

        x = tf.keras.layers.Conv2D(
            filter,
            kernel_size,
            strides=2,
            activation=activation,
            padding="same",
            name=name + "_conv2",
        )(x)

        x = tf.keras.layers.BatchNormalization(name=name + "_bn2")(x)

        compress_block = tf.keras.Model(inputs=inputs, outputs=x, name=name)

        return x, compress_block

    def expand_block(self, inputs, filter, kernel_size, activation, name):
        x = inputs

        x = tf.keras.layers.Conv2DTranspose(
            filter,
            kernel_size,
            strides=2,
            activation=activation,
            padding="same",
            name=name + "_conv1T",
        )(x)

        x = tf.keras.layers.BatchNormalization(name=name + "_bn1")(x)

        x = tf.keras.layers.Conv2DTranspose(
            filter,
            kernel_size,
            activation=activation,
            padding="same",
            name=name + "_conv2T",
        )(x)

        x = tf.keras.layers.BatchNormalization(name=name + "_bn2")(x)

        expand_block = tf.keras.Model(inputs=inputs, outputs=x, name=name)

        return x, expand_block

    def bottleneck(self, inputs):
        x = inputs

        x = tf.keras.layers.Flatten(name="encode_flatten")(x)

        x = tf.keras.layers.Dense(
            self.bottleneck_shape, activation="relu", name="encode_dense"
        )(x)

        x = tf.keras.layers.BatchNormalization(name="encode_bn")(x)

        mu = tf.keras.layers.Dense(self.latent_dim, name="mu")(x)

        sigma = tf.keras.layers.Dense(self.latent_dim, name="sigma")(x)

        z = Sampling()([mu, sigma])

        bottleneck = tf.keras.Model(
            inputs=inputs, outputs=[mu, sigma, z], name="bottleneck"
        )

        return mu, sigma, z, bottleneck

    def _encoder(self, inputs, filters, kernel_size, activation, name):
        x = inputs

        for filter in filters:
            x, compress_block = self.compress_block(
                inputs=x,
                filter=filter,
                kernel_size=kernel_size,
                activation=activation,
                name=name + "_compress_block" + str(filter),
            )

        self.conv_shape = x.shape

        mu, sigma, z, bottleneck = self.bottleneck(x)

        encoder = tf.keras.Model(inputs=inputs, outputs=[mu, sigma, z], name=name)

        return mu, sigma, z, encoder

    def _decoder(self, inputs, filters, kernel_size, activation="relu", name="decoder"):
        x = inputs

        units = tf.math.reduce_prod(self.conv_shape[1:])

        x = tf.keras.layers.Dense(units, activation=activation, name=name + "_dense")(x)
        x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)

        x = tf.keras.layers.Reshape(self.conv_shape[1:], name=name + "_reshape")(x)

        for filter in reversed(filters):
            x, expand_block = self.expand_block(
                inputs=x,
                filter=filter,
                kernel_size=kernel_size,
                activation=activation,
                name=name + "_expand_block" + str(filter),
            )

        x = tf.keras.layers.Conv2DTranspose(
            self.input_shape[-1],
            kernel_size,
            activation="sigmoid",
            padding="same",
            name=name + "_convT",
        )(x)

        decoder = tf.keras.Model(inputs=inputs, outputs=x, name=name)

        return x, decoder

    def _vae(self, inputs, filters, kernel_size, activation, name):
        mu, sigma, z, encoder = self._encoder(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            name=name + "_encoder",
        )

        x, decoder = self._decoder(
            inputs=z,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            name=name + "_decoder",
        )

        vae = tf.keras.Model(inputs=inputs, outputs=x, name=name)

        vae.add_loss(self.kl_reconstruction_loss(mu, sigma))

        return encoder, decoder, vae

    def kl_reconstruction_loss(self, mu, sigma):
        kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)

        kl_loss = tf.reduce_mean(kl_loss)

        kl_loss *= -0.5

        return kl_loss

    def build(
        self,
        filters=None,
        kernel_size=None,
        activation="relu",
        bottleneck_shape=16,
        latten_dim=2,
    ):
        self.bottleneck_shape = bottleneck_shape
        self.latent_dim = latten_dim

        inputs = tf.keras.Input(shape=self.input_shape)

        self.encoder, self.decoder, self.vae = self._vae(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            name="vae",
        )

    def summary(self):
        self.vae.summary()

    def compile(self, optimizer="adam", loss="binary_crossentropy", metrics=None):
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.loss = tf.keras.losses.get(loss)
        if metrics is not None:
            self.metrics = [
                tf.keras.metrics.get(metric) for metric in metrics
            ]
        else:
            self.metrics = None


    def encode(self, x, get_params=False):
        mu, sigma, z = self.encoder(x)
        if get_params:
            return mu, sigma, z
        return z
    
    def decode(self, z):
        return self.decoder(z)
    

    def fit(
        self,
        x,
        epochs=10,
        batch_size=32,
        show_image=False,
        show_interval=10,
        terminal=False,
    ):
        
        random_vector_for_generation = tf.random.normal(shape=[16, self.latent_dim])

        images = []

        self.fig, axs = plt.subplots(4, 4, figsize=(16, 16))

        self.axs = axs.flatten()

        img = None  

        for epoch in range(epochs):

            for step in range(x.shape[0] // batch_size):

                batch = x[step * batch_size : (step + 1) * batch_size]

                with tf.GradientTape() as tape:
                    reconstructed = self.vae(batch)

                    flatten_input = tf.keras.layers.Flatten()(batch)

                    flatten_reconstructed = tf.keras.layers.Flatten()(reconstructed)

                    loss_1 = self.loss(flatten_input, flatten_reconstructed) * (self.input_shape[0] * self.input_shape[1])

                    loss_2 = sum(self.vae.losses)

                    loss = loss_1 + loss_2

                gradients = tape.gradient(loss, self.vae.trainable_variables)

                self.optimizer.apply_gradients(
                    zip(gradients, self.vae.trainable_variables)
                )

                generated_image = self.decoder(random_vector_for_generation)

                display_image = self.show_image_progress(epoch, step, generated_image, loss)
                images.append(display_image)


                if step % show_interval == 0:
                    print(f"Epoch: {epoch} Step: {step} Regularized loss: {loss_1:3f} KL loss: {loss_2}", end=" ")

                    # print loss metrics if any
                    if self.metrics is not None:
                        for metric in self.metrics:
                            metric.reset_states()
                            metric.update_state(batch, reconstructed)
                            print(f"{metric.name}: {metric.result():3f}", end=" ")

                    print()

                    if show_image:
                        # check if it's jupyter notebook or terminal
                        if not terminal:  # True if it's jupyter notebook
                            display_fn(display_image, clear=True)
                        else:
                            im = np.array(display_image)
                            if img is None:
                                img = plt.imshow(im)
                            else:
                                img.set_data(im)
                            plt.pause(0.2)
                            plt.draw()

        return images


    def plt2arr(self, fig, draw=True):
        """Convert a Matplotlib figure to a numpy numpy array"""

        if draw:
            fig.canvas.draw()

        rgba_buffer = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(rgba_buffer, dtype=np.uint8).reshape(h, w, 4)
        return buf

    def show_image_progress(self, epoch, step, generated_image, loss):

        fig = self.fig
        axs = self.axs

        generated_image = generated_image.numpy()

        for i, ax in enumerate(axs):
            
            img = generated_image[i]*255

            img = img.astype(np.uint8)

            ax.imshow(img)

            ax.axis("off")

        self.fig.suptitle(f"Epoch: {epoch} Step: {step} Loss: {loss:3f}", fontsize=32)

        array = self.plt2arr(fig)

        image = Image.fromarray(array)
        image = image.resize((512, 512))
        return image


    def make_annime(self, images, filename, duration):
        """
        images:
            the list of images to make a gif from

        filename:
            the name of the gif file

        duration:
            Duration of the gif/video

        """
        extentention = filename.split(".")[-1]

        if extentention == "gif":
            imageio.mimwrite(filename, images, duration=duration)
        elif extentention == "mp4":
            fps = len(images) / duration
            fps = max(int(fps), 1)
            imageio.mimwrite(filename, images, fps=fps)
        else:
            raise ValueError("Invalid Extension (Only gif/mp4 are acceptable)")