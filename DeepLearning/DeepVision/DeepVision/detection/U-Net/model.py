import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class UNet:
    def __init__(self, input_shape, num_classes, pre_process=True):
        """
        Args:
            input_shape: tuple of 3 integers, dimensions of input image

            num_classes: integer, number of classes

            pre_process: boolean, whether to pre-process the input or not
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_process = pre_process

    def __preprocess(self, inputs):
        x = inputs

        x = tf.keras.layers.experimental.preprocessing.Resizing(height=512, width=512)(x)

        x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.)(x)

        return x

    def __conv_block(self, inputs, filter, activation, name=""):
        x = inputs

        x = tf.keras.layers.Conv2D(
            filter, 3, padding="same", activation=activation, name=name + "_conv1"
        )(x)

        x = tf.keras.layers.Conv2D(
            filter, 3, padding="same", activation=activation, name=name + "_conv2"
        )(x)

        return x

    def __encoder_block(self, inputs, filter, activation, dropout, name=""):
        x = inputs

        skip = self.__conv_block(x, filter, activation, name=name)

        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=name + "_pool")(skip)

        x = tf.keras.layers.Dropout(dropout, name=name + "_dropout")(x)

        return x, skip

    def __encoder(self, inputs, filters, activation, dropout):
        x = inputs

        skips = []

        for i, filter in enumerate(filters):
            x, skip = self.__encoder_block(
                x, filter, activation, dropout, name="EncoderBlock_" + str(i)
            )

            skips.append(skip)

        return x, skips

    def __bottle_neck(self, inputs, encoding_depth, activation):
        x = inputs

        x = self.__conv_block(x, encoding_depth, activation, "BottleNeck")

        return x

    def __decoder_block(self, inputs, skip, filter, activation, dropout, name=""):
        x = inputs

        x = tf.keras.layers.Conv2DTranspose(
            filter, 2, strides=(2, 2), padding="same", name=name + "_conv_transpose"
        )(x)

        x = tf.keras.layers.Concatenate(name=name + "_concat")([skip, x])

        x = tf.keras.layers.Dropout(dropout, name=name + "_dropout")(x)

        x = self.__conv_block(x, filter, activation, name=name)

        return x

    def __decoder(self, inputs, skips, filters, activation, dropout):
        x = inputs

        for i, filter in enumerate(filters):
            x = self.__decoder_block(
                x,
                skips[-i - 1],
                filter,
                activation,
                dropout,
                name="DecoderBlock_" + str(i),
            )

        return x

    def build(self, encoding_depth, filters, activation="relu", dropout=0.3):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name="input")

        if self.pre_process:
            x = self.__preprocess(inputs)
        else:
            x = inputs

        x, skips = self.__encoder(x, filters, activation, dropout)

        x = self.__bottle_neck(x, encoding_depth, activation)

        x = self.__decoder(x, skips, filters[::-1], activation, dropout)

        outputs = tf.keras.layers.Conv2D(
            self.num_classes,
            1,
            padding="same",
            activation="softmax",
            name="output",
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="U-Net")

        self.unet = model
    
    def create_mask(self, pred_mask):
        '''
        Creates the segmentation mask by getting the channel with the highest probability. Remember that we
        have 3 channels in the output of the UNet. For each pixel, the predicition will be the channel with the
        highest probability.
        '''
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0].numpy()

    def predict(self, image):
        '''
        Feeds an image to a model and returns the predicted mask.
        '''

        image = np.reshape(image,(1, image.shape[0], image.shape[1], image.shape[2]))
        pred_mask = self.unet.predict(image)
        pred_mask = self.create_mask(pred_mask)

        return pred_mask 
    
    def display(self, images ,titles=["Image", "True Mask"], display_string=None):
        '''displays a list of images/masks'''


        plt.figure(figsize=(10, 10))

        for i in range(len(images)):
            plt.subplot(1, len(images), i+1)
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
            if display_string and i == 1:
                plt.xlabel(display_string, fontsize=12)
            img_arr = tf.keras.preprocessing.image.array_to_img(images[i])
            plt.imshow(img_arr)
        
        plt.show()
    
    
