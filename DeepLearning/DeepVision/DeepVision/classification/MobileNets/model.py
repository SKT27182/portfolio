import tensorflow as tf
from DeepVision.blocks.blocks import MobileNetV1

class MobileNet:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = tf.keras.Sequential()
        self.final_shape = (224, 224)

        # add input layer
        self.model.add(tf.keras.layers.Input(shape=self.input_shape))


    def __preprocess(self, inputs):

        x = inputs

        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(x)

        prepocess = tf.keras.Model(inputs=inputs, outputs=x, name="Preprocess")

        self.model.add(prepocess)

        return x
    
    def __resolution_reduction(self, inputs, rho):

        x = inputs

        x = tf.keras.layers.experimental.preprocessing.Resizing(
            int(self.final_shape[0]*rho), int(self.final_shape[1]*rho)
        )(x)

        resolution_reduction = tf.keras.Model(
            inputs=inputs, outputs=x, name="Resolution_Reduction"
        )

        self.model.add(resolution_reduction)

        return x
    
    def __mobilenet_top(self, inputs, alpha=1.0):

        filters = int(32 * alpha)

        x = inputs

        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=3, strides=2, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        top = tf.keras.Model(inputs=inputs, outputs=x, name="MobileNet_Top")

        self.model.add(top)

        return x

    
    def __depthwise_blocks(self, inputs, alpha=1.0, depth_multiplier=1.0):

        x = inputs

        filters = 64

        for i in range(6):

            if i%2 == 0:
                strides = 1
            else:
                strides = 2

            filters = filters*strides

            depthwise_block = MobileNetV1(
                filters=filters, kernel_size=3, strides=strides, alpha=alpha, name=f"{i+1}"
            )

            x, block = depthwise_block(x)

            self.model.add(block)

        for i in range(5):
            depthwise_block = MobileNetV1(
                filters=512, kernel_size=3, strides=1, alpha=alpha, name=f"{i+7}"
            )

            x, block = depthwise_block(x)

            self.model.add(block)

        for i in range(2):
            depthwise_block = MobileNetV1(
                filters=1024, kernel_size=3, strides=2 if i==0 else 1, alpha=alpha, name=f"{i+12}"
            )

            x, block = depthwise_block(x)

            self.model.add(block)

        return x
    
    def __classifier(self, inputs, alpha=1.0):

        x = inputs

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Reshape((1, 1, int(1024 * alpha)))(x)

        x = tf.keras.layers.Conv2D(
            filters=self.output_shape, kernel_size=1, strides=1, padding="same"
        )(x)

        x = tf.keras.layers.Flatten()(x)

        classifier = tf.keras.Model(
            inputs=inputs, outputs=x, name="MobileNet_Classifier"
        )

        self.model.add(classifier)

        return x
    
    def build(self, alpha=1.0, rho=1, pre_process=True):

        inputs = tf.keras.layers.Input(shape=self.input_shape)

        if pre_process:
            x = self.__preprocess(inputs)
        else:
            x = inputs

        x = self.__resolution_reduction(x, rho=rho)

        x = self.__mobilenet_top(x, alpha=alpha)

        x = self.__depthwise_blocks(x, alpha=alpha)

        x = self.__classifier(x, alpha=alpha)

        return self.model