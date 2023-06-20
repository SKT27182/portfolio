import gradio as gr
from model import NeuralStyleTransfer
import tensorflow as tf
from keras import backend as K
import numpy as np


def change_dtype_inputs(
    n_style_layers,
    n_content_layers,
    epochs,
    learning_rate,
    steps_per_epoch,
    style_weight,
    content_weight,
    var_weight,
):
    return (
        int(n_style_layers),
        int(n_content_layers),
        int(epochs),
        float(learning_rate),
        int(steps_per_epoch),
        float(style_weight),
        float(content_weight),
        float(var_weight),
    )


def fit_style_transfer(
    style_image,
    content_image,
    extractor,
    n_style_layers,
    n_content_layers,
    epochs,
    learning_rate,
    steps_per_epoch,
    style_weight,
    content_weight,
    var_weight,
):
    """
    Fit the style transfer model to the content and style images.

    Parameters
    ----------

    style_image: str
        The path to the style image.

    content_image: str
        The path to the content image.

    extractor: str
        The name of the feature extractor to use. Options are
        "inception_v3", "vgg19", "resnet50", and "mobilenet_v2".

    n_style_layers: int
        The number of layers to use for the style loss.

    n_content_layers: int
        The number of layers to use for the content loss.

    epochs: int
        The number of epochs to train the model for.

    learning_rate: float
        The learning rate to use for the Adam optimizer.

    steps_per_epoch: int
        The number of steps to take per epoch.

    style_weight: float
        The weight to use for the style loss.

    content_weight: float
        The weight to use for the content loss.

    var_weight: float
        The weight to use for the total variation loss.

    Returns
    -------
    display_image: np.array
    """

    (
        n_style_layers,
        n_content_layers,
        epochs,
        learning_rate,
        steps_per_epoch,
        style_weight,
        content_weight,
        var_weight,
    ) = change_dtype_inputs(
        n_style_layers,
        n_content_layers,
        epochs,
        learning_rate,
        steps_per_epoch,
        style_weight,
        content_weight,
        var_weight,
    )

    model = NeuralStyleTransfer(
        style_image=style_image,
        content_image=content_image,
        extractor=extractor,
        n_style_layers=n_style_layers,
        n_content_layers=n_content_layers,
    )

    style_image = model.style_image
    content_image = model.content_image

    content_and_style_layers = model.get_output_layers()

    # build the model with the layers we need to extract the features from
    K.clear_session()
    model.build(content_and_style_layers)

    style_features = model.get_features(style_image, type="style")
    content_features = model.get_features(content_image, type="content")

    optimizer = tf.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=100, decay_rate=0.50
        )
    )

    generated_image = tf.cast(content_image, tf.float32)
    generated_image = tf.Variable(generated_image)

    step = 0

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            losses = model._update_image_with_style(
                generated_image,
                style_features,
                content_features,
                style_weight,
                content_weight,
                optimizer,
                var_weight,
            )

            display_image = model.tensor_to_image(generated_image)

            step += 1

            style_loss, content_loss, var_loss = losses

            yield np.array(display_image), style_loss, content_loss, var_loss, epoch, step



def main():
    content_image = gr.Image(type="filepath", label="Content Image")
    style_image = gr.Image(type="filepath", label="Style Image")

    extractor = gr.Dropdown(
        ["inception_v3", "vgg19", "resnet50", "mobilenet_v2"],
        label="Feature Extractor",
        value="inception_v3",

    )

    n_content_layers = gr.Slider(
        1,
        5,
        value=3,
        step=1,
        label="Content Layers",
    )

    n_style_layers = gr.Slider(
        1,
        5,
        value=3,
        step=1,
        label="Style Layers",
    )

    epochs = gr.Slider(2, 20, value=4, step=1, label="Epochs")

    learning_rate = gr.Slider(1, 50, value=20, step=1, label="Learning Rate")

    steps_per_epoch = gr.Slider(
        1,
        20,
        value=5,
        step=1,
        label="Steps Per Epoch",
    )

    style_weight = gr.Slider(
        0,
        1,
        value=0.1,
        label="Style Weight",
    )

    content_weight = gr.Slider(
        0,
        1,
        value=1,
        label="Content Weight",
    )

    var_weight = gr.Slider(
        0,
        0.8,
        value=0,
        label="Variation Weight",
    )

    inputs = [
        style_image,
        content_image,
        extractor,
        n_style_layers,
        n_content_layers,
        epochs,
        learning_rate,
        steps_per_epoch,
        style_weight,
        content_weight,
        var_weight,
    ]

    examples = [
        [
            "examples/van_gogh.jpg",
            "examples/scene.jpg",
            "inception_v3",
            5,
            5,
            10,
            40,
            10,
            0.1,
            1,
            0.0,
        ],
        [
            "examples/painting.jpg",
            "examples/swan.jpg",
            "vgg19",
            5,
            5,
            10,
            40,
            10,
            0.1,
            1,
            0.0,
        ]
    ]

    output_image = gr.Image(type="numpy", label="Output Image")

    style_loss = gr.Number(label="Current Style Loss")

    content_loss = gr.Number(label="Current Content Loss")

    var_loss = gr.Number(label="Current Total Variation Loss")

    curr_epoch = gr.Number(label="Current Epoch")

    curr_step = gr.Number(label="Current Step")

    title = "Neural Style Transfer"

    description = """### This app uses a neural network to transfer the style of one image to another. \n### The `style image` is the image whose style you want to transfer, and the `content image` is the image you want to transfer the style to. \n### The `feature extractor` is the neural network used to extract the features from the images. \n### The number of `style layers` and `content layers` are the number of layers in the feature extractor used to extract the style and content features respectively. \n### The `epochs`, `learning_rate`, `steps_per_epoch`, `style_weight`, `content_weight`, and `total_variation_weight` are all **hyperparameters** that affect the style transfer process. \n### The style weight controls how much the style image affects the output image, the content weight controls how much the content image affects the output image, and the variation weight controls how much the total variation of the output image affects the output image. The total variation of an image is the sum of the absolute differences for neighboring pixel-values in the image. The total variation loss is used to smooth the output image. The higher the variation weight, the smoother the output image will be."""



    outputs = [output_image, style_loss, content_loss, var_loss, curr_epoch, curr_step]

    interface = gr.Interface(
        fn=fit_style_transfer,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description,
        examples=examples,
        theme='gradio/monochrome' 
        
    )

    interface.queue().launch(server_name="0.0.0.0", server_port=7860)

main()
