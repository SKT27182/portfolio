import tensorflow as tf
from DeepVision.blocks.blocks import MobileNetV2

EfficientNetB0_filters = [16, 24, 40, 80, 112, 192, 320]  # filters

Expansion_Ratio = [1, 6, 6, 6, 6, 6, 6]  # expansion ratio
EfficientNetB0_layers = [1, 2, 2, 3, 3, 4, 1]  # layers of each block
EfficientNetB0_kernel = [3, 3, 5, 3, 5, 5, 3]  # kernel size
se_ratio = 1 / 16

alpha = 1.2  # depth multiplier (layers in each block)
beta = 1.1  # width multiplier (filter)
gamma = 1.15  # resolution multiplier (input size)


phi_values = {
    0: 224,
    1: 240,
    2: 260,
    3: 300,
    4: 380,
    5: 456,
    6: 528,
    7: 600,
}  # resolution scaling


strides = [
    1,  # stride
    2,  # stride
    2,  # stride
    2,  # stride
    1,  # stride
    2,  # stride
    1,  # stride
]


class EfficientNet:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = tf.keras.Sequential()

        # add input layer
        self.model.add(tf.keras.layers.Input(shape=self.input_shape))

    
    def __preprocess(self, inputs):

        x = inputs

        # normalize between -1 and 1

        x = tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0, name="normalization")(x)

        # add to the model
        prepocess = tf.keras.Model(inputs=inputs, outputs=x, name="Preprocess")

        self.model.add(prepocess)

        return x
    
    def __resolution_scale(self, inputs):

        x = inputs

        x = tf.keras.layers.experimental.preprocessing.Resizing(self.final_shape[0], self.final_shape[1], interpolation="bilinear")(x)

        resolution_scale = tf.keras.Model(inputs=inputs, outputs=x, name="resolution_scale")

        self.model.add(resolution_scale)

        return x

    def __build_top(self, inputs):
        x = inputs

        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, padding="same"
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)

        top = tf.keras.Model(inputs=inputs, outputs=x, name="top")

        self.model.add(top)

        return x

    def __build_classifier(self, inputs):
        x = inputs

        x = tf.keras.layers.Conv2D(
            filters=1280, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(self.output_shape, activation="softmax")(x)

        classifier = tf.keras.Model(inputs=inputs, outputs=x, name="classifier")

        self.model.add(classifier)

        return x

    def __build_repeat_block(
        self,
        inputs,
        filters,
        kernel,
        expansion,
        se_ratio,
        repeat_layer,
        downsample=False,
        name="",
    ):
        x = inputs

        if downsample:
            x, block = MobileNetV2(
                filters,
                kernel,
                strides=2,
                expansion=expansion,
                se_ratio=se_ratio,
                name=name + "_1",
            )(x)
        else:
            x, block = MobileNetV2(
                filters,
                kernel,
                strides=1,
                activation="swish",
                expansion=expansion,
                se_ratio=se_ratio,
                name=name + "_1",
            )(x)

        self.model.add(block)

        for i in range(1, repeat_layer):
            x, block = MobileNetV2(
                filters,
                kernel,
                strides=1,
                activation="swish",
                expansion=expansion,
                se_ratio=se_ratio,
                name=name + "_" + str(i + 1),
            )(x)
            self.model.add(block)

        return x

    def __layers(self, phi):
        
        multiplier = alpha ** phi

        layers_ = EfficientNetB0_layers

        return [int(multiplier * i) for i in layers_]
    
    
    def __filters(self, phi):

        multiplier = beta ** phi

        filters_ = EfficientNetB0_filters

        return [int(multiplier * i) for i in filters_]
    

    def __resolution(self, phi):

        return (phi_values[phi], phi_values[phi], 3)
    

    def build(self, phi=0, pre_process=True):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        x = inputs

        filters_ = self.__filters(phi)

        layers_ = self.__layers(phi)

        self.final_shape = self.__resolution(phi)

        if pre_process:
            x = self.__preprocess(x)

        x = self.__resolution_scale(x)

        x = self.__build_top(x)

        for i in range(len(filters_)):
            x = self.__build_repeat_block(
                x,
                filters_[i],
                EfficientNetB0_kernel[i],
                Expansion_Ratio[i],
                se_ratio,
                layers_[i],
                downsample=(strides[i] == 2),
                name="block" + str(i + 1),
            )

        x = self.__build_classifier(x)

        return self.model