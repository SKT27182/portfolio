import tensorflow as tf
from DeepVision.blocks.blocks import ResidualIdenticalBlock, ResidualBottleNeckBlock, ResPlainBlock

class ResNet:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.block = None
        self.repeate_block = None
        self.blocks = {
            "ResidualIdenticalBlock": ResidualIdenticalBlock,
            "ResidualBottleNeckBlock": ResidualBottleNeckBlock,
            "ResidualPlainBlock": ResPlainBlock,
        }

    def __resnet_top(self, inputs):
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=7, strides=2, padding="same"
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

        resnet_top = tf.keras.Model(inputs=inputs, outputs=x, name="ResNet_top")

        return x, resnet_top

    def __resnet_bottom(self, inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=self.output_shape)(x)
        x = tf.keras.layers.Activation("softmax")(x)

        resnet_bottom = tf.keras.Model(inputs=inputs, outputs=x, name="ResNet_bottom")

        return x, resnet_bottom
        

    def __make_block(self, inputs, filters, layers, strides=1, name=""):

        res_layer = tf.keras.Sequential(name="Block_" + name)
        x, block = self.block(
            filters=filters, kernel_size=3, strides=strides, name="SubBlock_1"
        )(inputs)

        res_layer.add(block)

        for i in range(1, layers):
            x , block = self.block(filters=filters, kernel_size=3, name="SubBlock_" + str(i+1))(x)
            res_layer.add(block)
            
        return x, res_layer

    def build(self, name="ResNet"):

        model = tf.keras.Sequential(name=name)

        inputs = tf.keras.Input(shape=self.input_shape)

        # Top of the ResNet model
        x, resnet_top = self.__resnet_top(inputs)

        model.add(resnet_top)

        # Body of the ResNet model

        # Block-1
        x, block = self.__make_block(inputs = x, filters=64, layers=self.repeate_block[0], name="1")

        model.add(block)

        # Block-2 to 4
        for i in range(1, len(self.repeate_block)):
            x, block = self.__make_block(inputs = x,
                filters=64 * (2 ** i), layers=self.repeate_block[i], strides=2, name=str(i + 1)
            )
            model.add(block)

        # Bottom of the ResNet model
        x, resnet_bottom = self.__resnet_bottom(x)

        model.add(resnet_bottom)

        return model

    def resnet18(self, block, repeate_block):
        self.block = self.blocks[block]
        self.repeate_block = repeate_block
        return self.build(name="ResNet18")

    def resnet34(self, block, repeate_block):
        self.block = self.blocks[block]
        self.repeate_block = repeate_block
        return self.build(name="ResNet34")

    def resnet50(self, block, repeate_block):
        self.block = self.blocks[block]
        self.repeate_block = repeate_block
        return self.build(name="ResNet50")

    def resnet101(self, block, repeate_block):
        self.block = self.blocks[block]
        self.repeate_block = repeate_block
        return self.build(name="ResNet101")

    def resnet152(self, block, repeate_block):
        self.block = self.blocks[block]
        self.repeate_block = repeate_block
        return self.build(name="ResNet152")
    