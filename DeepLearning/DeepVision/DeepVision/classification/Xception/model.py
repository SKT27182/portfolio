import tensorflow as tf

class Xception:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = tf.keras.Sequential()

        # add input layer
        self.model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name="Input"))

    def __preprocess(self, inputs):

        x = inputs

        # if input height and width is smaller than 299, then resize it to 299
        if (x.shape[1] < 299 or x.shape[2] < 299):
            x = tf.keras.layers.experimental.preprocessing.Resizing(
                height=299, width=299
            )(x)

        # if height and width are not equal then make a square with smaller dimension
        if (x.shape[1] != x.shape[2]):
            x = tf.keras.layers.experimental.preprocessing.CenterCrop(
                height=min(x.shape[1], x.shape[2]), width=min(x.shape[1], x.shape[2])
            )(x)

        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(x)

        prepocess = tf.keras.models.Model(inputs, x, name="Preprocess")

        # add preprocess model to the main model

        self.model.add(prepocess)

        return x
    
    def __entry_flow(self, inputs):

        x = inputs

        x = tf.keras.layers.Conv2D( filters=32, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D( filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        previous_block_activation = x

        for size in [128, 256, 728]:

            if size != 128:
                x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters=size, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters=size, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

            residual = tf.keras.layers.Conv2D(filters=size, kernel_size=1, strides=2, padding='same')(previous_block_activation)
            residual = tf.keras.layers.BatchNormalization()(residual)

            x = tf.keras.layers.add([x, residual])
            previous_block_activation = x

        entry_flow = tf.keras.models.Model(inputs, x, name="Entry_Flow")

        # add entry flow model to the main model

        self.model.add(entry_flow)

        return x
    
    def __middle_flow(self, inputs):

        x = inputs

        for i in range(8):

            previous_block_activation = x

            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.add([x, previous_block_activation])

        middle_flow = tf.keras.models.Model(inputs, x, name="Middle_Flow")

        # add middle flow model to the main model

        self.model.add(middle_flow, )

        return x
    
    def __exit_flow(self, inputs):

        x = inputs

        previous_block_activation = x

        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        residual = tf.keras.layers.Conv2D(filters=1024, kernel_size=1, strides=2, padding='same')(previous_block_activation)
        residual = tf.keras.layers.BatchNormalization()(residual)

        x = tf.keras.layers.add([x, residual])

        x = tf.keras.layers.SeparableConv2D(filters=1536, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.SeparableConv2D(filters=2048, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        exit_flow = tf.keras.models.Model(inputs, x , name="Exit_Flow")

        # add exit flow model to the main model

        self.model.add(exit_flow)

        return x
    
    def __classifier(self, inputs):

        x = inputs

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(units=2048, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)

        outputs = tf.keras.layers.Dense(units=self.output_shape, activation='softmax')(x)

        classifier = tf.keras.models.Model(inputs, outputs , name="Classifier")

        # add classifier model to the main model

        self.model.add(classifier)

        return x
    
    def build(self, pre_process=True):

        inputs = tf.keras.Input(shape=self.input_shape)

        if pre_process:
            x = self.__preprocess(inputs)
        else:
            x = inputs
            
        x = self.__entry_flow(x)
        x = self.__middle_flow(x)
        x = self.__exit_flow(x)

        x = self.__classifier(x)

        return self.model
