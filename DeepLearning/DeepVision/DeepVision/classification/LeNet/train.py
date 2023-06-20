import tensorflow as tf
import argparse
from DeepVision.classification.LeNet.model import LeNet
from DeepVision.utils.data import Datasets
from DeepVision.utils.helper import *


def lenet_preprocess(x, output_shape=(32, 32, 1)):

    """

    Add padding or cropping to the input tensor to make it of size output_shape (which is input shape for the LeNet models)

    preprocess the input tensor

    mean = 0
    variance = 1

    """

    x = tf.image.resize_with_pad(x, output_shape[0], output_shape[1])

    x = tf.keras.layers.Normalization(
        axis=-1, mean=0, variance=1, invert=False, name="input"
    )(x)

    return x


# output shape of the preprocessing function i.e. the input shape of the LeNet models
output_shape = {
    "LeNet_5_exact": (32, 32, 1),
    "LeNet_5_mod_1": (32, 32, 1),
    "LeNet_5_mod_2": (28, 28, 1),
}


# make an instance of the LeNet class
def load_model(args):
    if args.model == "LeNet_5_exact":
        model = LeNet(input_shape=output_shape["LeNet_5_exact"], output_shape=args.output_shape).lenet_5_exact()
    elif args.model == "LeNet_5_mod_1":
        model = LeNet(input_shape=output_shape["LeNet_5_mod_1"], output_shape=args.output_shape).lenet_5_mod_1()
    elif args.model == "LeNet_5_mod_2":
        model = LeNet(input_shape=output_shape["LeNet_5_mod_2"], output_shape=args.output_shape).lenet_5_mod_2()

    return model


def main(args):
    model = load_model(args)

    # model_summary_only only
    if args.summary_only:
        model.summary()
        return

    # if args.architecture:
    #     tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True)


    if args.train:

        dataset = Datasets().load_dataset(args.dataset)

        (x_train, y_train), (x_test, y_test) = dataset

        if args.preprocessing:
            x_train = lenet_preprocess(x_train, output_shape[args.model])
            x_test = lenet_preprocess(x_test, output_shape[args.model])

        # reshape y to (batch_size, 1, 1, 10)
        if args.model == "LeNet_5_exact":
            y_train = tf.reshape(y_train, (y_train.shape[0], 1, 1, 10))
            y_test = tf.reshape(y_test, (y_test.shape[0], 1, 1, 10))

        model.compile(
            optimizer=optimizers(args.optimizer, args.learning_rate),
            loss=losses(args.loss),
            metrics=[metrics(args.metrics)],
        )

        model.fit(
            x_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_test, y_test),
        )


def arg_parse():
    args = argparse.ArgumentParser(add_help=True)

    args.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )

    args.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset to use for training"
    )

    args.add_argument(
        "--model", type=str, default="LeNet_5_mod_2", help="Model to use for training"
    )

    args.add_argument(
        "--preprocessing",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="Whether to use preprocessing or not",
    )

    args.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use for training"
    )

    args.add_argument(
        "--loss",
        type=str,
        default="categorical_crossentropy",
        help="Loss function to use for training",
    )

    args.add_argument(
        "--metrics", type=str, default="accuracy", help="Metrics to use for training"
    )

    args.add_argument(
        "--summary_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args.add_argument(
        "--architecture",
        type=bool,
        default=True,
        help="Whether to print model architecture or not",
    )

    args.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train or not",
    )

    args.add_argument(
        "--input_shape",
        type=tuple,
        default=(32, 32, 3),
        help="Output shape of the model",
    )

    args.add_argument(
        "--output_shape",
        type=int,
        default=10,
        help="Output shape of the model",
    )

    args = args.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)
