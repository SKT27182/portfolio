import tensorflow as tf
from DeepVision.blocks.blocks import DenseBlock


DenseNet_Layers = {
    121: [6, 12, 24, 16],
    169: [6, 12, 32, 32],
    201: [6, 12, 48, 32],
    264: [6, 12, 64, 48],
}


class DenseNet:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.model = tf.keras.Sequential()

        # add input layer
        self.model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    def __preprocess(self, inputs):

        x = inputs

        x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(x)

        # normalize between -1 and 1

        x = tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0, name="normalization")(x)

        # add to the model
        prepocess = tf.keras.Model(inputs=inputs, outputs=x, name="Preprocess")

        self.model.add(prepocess)

        return x

    def _densenet_top(self, inputs):
        x = inputs
    
        x = tf.keras.layers.Conv2D(
            2 * self.k, kernel_size=7, strides=2, padding="same"
        )(x)
        
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

        top = tf.keras.Model(inputs=inputs, outputs=x, name="DenseNet_Top")

        # add it to the model
        self.model.add(top)

        return x

    def _densenet_bottom(self, inputs):
        x = inputs
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.output_shape, activation="softmax")(x)

        classifier = tf.keras.Model(
            inputs=inputs, outputs=x, name="DenseNet_Classifier" 
        )

        # add it to the model
        self.model.add(classifier)

        return x

    def _transition_layer(self, inputs, theta, name=None):
        x = inputs

        input_features = x.shape[-1]

        ouput_features = int(theta * input_features)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(ouput_features, 1, strides=1, padding="same")(x)

        x = tf.keras.layers.AveragePooling2D(2, 2)(x)

        transition = tf.keras.Model(
            inputs=inputs, outputs=x, name="Transition_Layer" + name
        )

        # add it to the model
        self.model.add(transition)

        return x

    def _dense_block(self, inputs, n_blocks, theta, name=None):
        x = inputs

        repeated_layers = DenseNet_Layers[self.size]

        for i in range(n_blocks - 1):
            dense_blocs = DenseBlock(
                repeated_layers[i], self.k, self.drop_rate, name=str(i)
            )

            x, block = dense_blocs(x)

            self.model.add(block)

            # add transition layer
            x = self._transition_layer(x, theta=theta, name=str(i))

        # add last dense block
        dense_blocs = DenseBlock(repeated_layers[-1], self.k, name=str(n_blocks - 1))

        x, block = dense_blocs(x)

        self.model.add(block)

        return x

    def build(self, k=32, drop_rate=0.2, size=121, theta=1,  pre_process=True):

        self.k = k
        self.drop_rate = drop_rate
        self.size = size
    
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # if preproc
        if pre_process:
            x = self.__preprocess(inputs)

        else:
            x = inputs
        
        x = self._densenet_top(x)

        x = self._dense_block(x, 4, theta)

        x = self._densenet_bottom(x)

        return self.model