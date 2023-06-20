import tensorflow as tf


class VGG:

    """
    VGG model

    VGG is a convolutional neural network architecture named after Visual Geometry Group from Oxford University. It was the first large-scale deep neural network to be trained on the ImageNet dataset.

    In this implementation, the model is trained on the CIFAR-10 dataset.

    input_shape:
        X: (None, 224, 224, 3)
        y: (None,  classes)

    output_shape:
        y: (None, classes)

    """

    def __init__(self, input_shape=(224, 224, 3), output_shape=1000):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __conv_bolck(
        self,
        name,
        input_tensor,
        filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        activation="relu",
    ):

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            name=name,
        )(input_tensor)

        return x

    def __vgg_block(self, name, input_tensor, conv_layer, ker_size):

        x = input_tensor

        for i in range(len(conv_layer)):
            ks = ker_size[i], ker_size[i]
            x = self.__conv_bolck(
                name=name + "_conv" + str(i + 1), input_tensor=x, filters=conv_layer[i], kernel_size=ks
            )

        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=2, name=name + "_MaxPool"
        )(x)

        return x

    def __dense_block(self, name, input_tensor, dense_layer=3):

        x = input_tensor

        for i in range(len(dense_layer)-1):
            x = tf.keras.layers.Dense(
                units=dense_layer[i], activation="relu", name=name  + str(i + 1)
            )(x)
            x = tf.keras.layers.Dropout(rate=0.5, name=name + "_Dropout" + str(i + 1))(x)

        x = tf.keras.layers.Dense(
            units=dense_layer[-1], activation="softmax", name=name  + str(i + 2)
        )(x)

        return x

    def vgg_a(self):

        input_img = tf.keras.Input(shape=self.input_shape, name="input")

        x = self.__vgg_block(
            input_tensor=input_img, conv_layer=[64],  name="Block1", ker_size=[3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[128],  name="Block2", ker_size=[3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[256, 256],  name="Block3", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512],  name="Block4", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512],  name="Block5", ker_size=[3,3]
        )

        x = tf.keras.layers.Flatten(name="Flatten")(x)

        x = self.__dense_block(name="FC", input_tensor=x, dense_layer=[4096, 4096, self.output_shape])

        model = tf.keras.Model(inputs=input_img, outputs=x, name="VGG_A")

        return model
    
    def vgg_a_lrn(self):
            
        input_img = tf.keras.Input(shape=self.input_shape, name="input")


        x = self.__conv_bolck(
            name="Block1_conv1", input_tensor=input_img, filters=64, kernel_size=(3,3)
        )

        # Adding Local Response Normalization

        x = tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name="LRN2")

        x = self.__vgg_block(
            input_tensor=x, conv_layer=[128],  name="Block2", ker_size=[3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[256, 256],  name="Block3", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512],  name="Block4", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512],  name="Block5", ker_size=[3,3]
        )

        x = tf.keras.layers.Flatten(name="Flatten")(x)

        x = self.__dense_block(name="FC", input_tensor=x, dense_layer=[4096, 4096, self.output_shape])

        model = tf.keras.Model(inputs=input_img, outputs=x, name="VGG_A")

        return model
        
    

    def vgg_b(self):

        input_img = tf.keras.Input(shape=self.input_shape, name="input")

        x = self.__vgg_block(
            input_tensor=input_img, conv_layer=[64, 64],  name="Block1", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[128, 128],  name="Block2", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[256, 256],  name="Block3", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512],  name="Block4", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512],  name="Block5", ker_size=[3,3]
        )

        x = tf.keras.layers.Flatten(name="Flatten")(x)

        x = self.__dense_block(name="FC", input_tensor=x, dense_layer=[4096, 4096, self.output_shape])

        model = tf.keras.Model(inputs=input_img, outputs=x, name="VGG_B")

        return model
    
    def vgg_c(self):

        input_img = tf.keras.Input(shape=self.input_shape, name="input")

        x = self.__vgg_block(
            input_tensor=input_img, conv_layer=[64, 64],  name="Block1", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[128, 128],  name="Block2", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[256, 256, 256],  name="Block3", ker_size=[3,3,1]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512, 512],  name="Block4", ker_size=[3,3,1]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512, 512],  name="Block5", ker_size=[3,3,1]
        )

        x = tf.keras.layers.Flatten(name="Flatten")(x)

        x = self.__dense_block(name="FC", input_tensor=x, dense_layer=[4096, 4096, self.output_shape])

        model = tf.keras.Model(inputs=input_img, outputs=x, name="VGG_C")

        return model
    
    def vgg_d(self):
            
        input_img = tf.keras.Input(shape=self.input_shape, name="input")

        x = self.__vgg_block(
            input_tensor=input_img, conv_layer=[64, 64],  name="Block1", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[128, 128],  name="Block2", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[256, 256, 256],  name="Block3", ker_size=[3,3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512, 512],  name="Block4", ker_size=[3,3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512, 512],  name="Block5", ker_size=[3,3,3]
        )

        x = tf.keras.layers.Flatten(name="Flatten")(x)

        x = self.__dense_block(name="FC", input_tensor=x, dense_layer=[4096, 4096, self.output_shape])

        model = tf.keras.Model(inputs=input_img, outputs=x, name="VGG_D")

        return model
    
    def vgg_e(self):
            
        input_img = tf.keras.Input(shape=self.input_shape, name="input")

        x = self.__vgg_block(
            input_tensor=input_img, conv_layer=[64, 64],  name="Block1", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[128, 128],  name="Block2", ker_size=[3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[256, 256, 256],  name="Block3", ker_size=[3,3,3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512, 512],  name="Block4", ker_size=[3,3,3,3]
        )
        x = self.__vgg_block(
            input_tensor=x, conv_layer=[512, 512, 512, 512],  name="Block5", ker_size=[3,3,3,3]
        )

        x = tf.keras.layers.Flatten(name="Flatten")(x)

        x = self.__dense_block(name="FC", input_tensor=x, dense_layer=[4096, 4096, self.output_shape])

        model = tf.keras.Model(inputs=input_img, outputs=x, name="VGG_E")

        return model
