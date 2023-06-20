import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


class NeuralStyleTransfer:
    def __init__(self, style_image, content_image, extractor, n_style_layers=5, n_content_layers=5):

        self.extractor = extractor
        # load the model
        if extractor == "inception_v3":
            self.feature_extractor = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet"
            )
        elif extractor == "vgg19":
            self.feature_extractor = tf.keras.applications.VGG19(
                include_top=False, weights="imagenet"
            )
        elif extractor == "resnet50":
            self.feature_extractor = tf.keras.applications.ResNet50(
                include_top=False, weights="imagenet"
            )
        elif extractor == "mobilenet_v2":
            self.feature_extractor = tf.keras.applications.MobileNetV2(
                include_top=False, weights="imagenet"
            )
        elif isinstance(extractor, tf.keras.Model):
            self.feature_extractor = extractor
        else:
            raise Exception("Features Extractor not found")

        # freeze the model
        self.feature_extractor.trainable = False

        # define the style and content depth
        self.n_style_layers = n_style_layers
        self.n_content_layers = n_content_layers

        self.style_image = self._load_img(style_image)
        self.content_image = self._load_img(content_image)

    def tensor_to_image(self, tensor):
        """converts a tensor to an image"""
        tensor_shape = tf.shape(tensor)
        number_elem_shape = tf.shape(tensor_shape)
        if number_elem_shape > 3:
            assert tensor_shape[0] == 1
            tensor = tensor[0]
        return tf.keras.preprocessing.image.array_to_img(tensor)

    def _load_img(self, image):
        max_dim = 512

        image = tf.io.read_file(image)
        image = tf.image.decode_image(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.convert_image_dtype(image, tf.float32)

        shape = tf.shape(image)[:-1]
        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]
        image = tf.image.convert_image_dtype(image, tf.uint8)

        return image


    def _preprocess_image(self, image):
        image = tf.cast(image, dtype=tf.float32)
        
        if self.extractor == "inception_v3":
            image = image / 127.5 - 1
        elif self.extractor == "vgg19":
            image = tf.keras.applications.vgg19.preprocess_input(image)
        elif self.extractor == "resnet50":
            image = tf.keras.applications.resnet50.preprocess_input(image)
        elif self.extractor == "mobilenet_v2":
            # scale pixel between -1 and 1
            image = image / 127.5 - 1
        return image

    def get_output_layers(self):
        # get all the layers which contain conv in their name
        all_layers = [
            layer.name
            for layer in self.feature_extractor.layers
            if "conv" in layer.name
        ]

        # define the style layers
        style_layers = all_layers[: self.n_style_layers]

        # define the content layers from second last layer
        content_layers = all_layers[-2: -self.n_content_layers - 2 : -1]

        content_and_style_layers = content_layers + style_layers

        return content_and_style_layers

    def build(self, layers_name):

        output_layers = [
            self.feature_extractor.get_layer(name).output for name in layers_name
        ]

        model = tf.keras.Model(self.feature_extractor.input, output_layers)

        self.feature_extractor = model

        return

    def _loss(self, target_img, features_img, type):
        """
        Calculates the loss of the style transfer

        target_img:
            the target image (style or content) features

        features_img:
            the generated image features (style or content)

        """

        loss = tf.reduce_mean(tf.square(features_img - target_img))

        if type == "content":
            return 0.5 * loss

        return loss

    def _gram_matrix(self, input_tensor):
        """
        Calculates the gram matrix and divides by the number of locations

        input_tensor:
            the output of the conv layer of the style image, shape = (batch_size, height, width, channels)

        """
        result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    def get_features(self, image, type):
        preprocess_image = self._preprocess_image(image)

        outputs = self.feature_extractor(preprocess_image)

        if type == "style":
            outputs = outputs[self.n_content_layers : ]
            features = [self._gram_matrix(style_output) for style_output in outputs]

        elif type == "content":
            features = outputs[ : self.n_content_layers]

        return features

    def _style_content_loss(
        self,
        style_targets,
        style_outputs,
        content_targets,
        content_outputs,
        style_weight,
        content_weight,
    ):
        """
        Calculates the total loss of the style transfer

        style_targets:
            the style features of the style image

        style_outputs:
            the style features of the generated image

        content_targets:
            the content features of the content image

        content_outputs:
            the content features of the generated image

        style_weight:
            the weight of the style loss

        content_weight:
            the weight of the content loss

        """

        # adding the loss of each layer
        style_loss = style_weight * tf.add_n(
            [
                self._loss(style_target, style_output, type="style")
                for style_target, style_output in zip(style_targets, style_outputs)
            ]
        )
        content_loss = content_weight * tf.add_n(
            [
                self._loss(content_target, content_output, type="content")
                for content_target, content_output in zip(
                    content_targets, content_outputs
                )
            ]
        )

        style_loss /= self.n_style_layers
        content_loss /= self.n_content_layers

        
        total_loss = style_loss + content_loss
        return total_loss, style_loss, content_loss

    def _grad_loss(
        self,
        generated_image,
        style_target,
        content_target,
        style_weight,
        content_weight,
        var_weight,
    ):
        """
        Calculates the gradients of the loss function with respect to the generated image

        generated_image:
            the generated image

        """

        with tf.GradientTape() as tape:
            style_features = self.get_features(generated_image, type="style")
            content_features = self.get_features(generated_image, type="content")
            loss, style_loss, content_loss = self._style_content_loss(
                style_target,
                style_features,
                content_target,
                content_features,
                style_weight,
                content_weight,
            )

            variational_loss= var_weight*tf.image.total_variation(generated_image)

            loss += variational_loss
        grads = tape.gradient(loss, generated_image)
        return grads, loss, [style_loss, content_loss, variational_loss]

    def _update_image_with_style(
        self,
        generated_image,
        style_target,
        content_target,
        style_weight,
        content_weight,
        optimizer,
        var_weight,
    ):
        grads, loss, loss_list = self._grad_loss(
            generated_image, style_target, content_target, style_weight, content_weight, var_weight
        )

        optimizer.apply_gradients([(grads, generated_image)])

        generated_image.assign(
            tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=255.0)
        )
        return loss_list