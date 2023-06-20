import tensorflow as tf
import argparse
from DeepVision.classification.ResNet.model import ResNet
from DeepVision.utils.data import Datasets
from DeepVision.utils.helper import *
import ast

class TupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(ast.literal_eval(values)))


def resnet_preprocess(x):

    """

    Add padding or cropping to the input tensor to make it of size output_shape (which is input shape for the LeNet models)

    preprocess the input tensor

    input_shape: it is the shape of the input of the model.

    """


    x = tf.keras.layers.experimental.preprocessing.Resizing(
        224, 224
    )(x)

    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(x)

    return x


resnet_repeat_block = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3],
}

resnet_models = {
    "resnet18": resnet_repeat_block["resnet18"],
    "resnet34": resnet_repeat_block["resnet34"],
    "resnet50": resnet_repeat_block["resnet50"],
    "resnet101": resnet_repeat_block["resnet101"],
    "resnet152": resnet_repeat_block["resnet152"],
}

resnet_blocks = {
    "identical" : "ResidualIdenticalBlock",
    "bottle_neck" : "ResidualBottleNeckBlock",
    "plain" : "ResidualPlainBlock"
}


# make an instance of the LeNet class
def load_model(args):
    resnet_model = ResNet(input_shape=args.input_shape, output_shape=args.output_shape)
    block = resnet_blocks[args.block]
    repeate_block = resnet_repeat_block[args.model]
    if args.model == "resnet18":
        model = resnet_model.resnet18( block=block, repeate_block=repeate_block)
    elif args.model == "resnet34":
        model = resnet_model.resnet34( block=block, repeate_block=repeate_block)
    elif args.model == "resnet50":
        model = resnet_model.resnet50( block=block, repeate_block=repeate_block)
    elif args.model == "resnet101":
        model = resnet_model.resnet101( block=block, repeate_block=repeate_block)
    elif args.model == "resnet152":
        model = resnet_model.resnet152( block=block, repeate_block=repeate_block)

    return model


def main(args):
    model = load_model(args)

    # model_summary_only only
    if args.summary_only:
        model.summary(expand_nested=True)

    # if args.architecture:
    #     tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True)


    if args.train:

        dataset = Datasets().load_dataset(args.dataset)

        (x_train, y_train), (x_test, y_test) = dataset

        if args.preprocessing:
            x_train = resnet_preprocess(x_train)
            x_test = resnet_preprocess(x_test)

        model.compile(
            optimizer=optimizers(args.optimizer, args.learning_rate),
            loss=losses(args.loss),
            metrics=[metrics(args.metrics)],
        )

        all_images = model.fit(
            x_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_test, y_test),
        )

        



def arg_parse():
    args = argparse.ArgumentParser(add_help=True)

    args.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args.add_argument("--batch_size", type=int, default=256, help="Batch size")

    args.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

    args.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset to use for training"
    )

    args.add_argument(
        "--model", type=str, default="resnet18", help="Model to use for training"
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
        "--train",
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
        "--input_shape",
        action=TupleAction,
        default=(224, 224, 3),
        help="Input shape of the model",
    )

    args.add_argument(
        "--output_shape",
        type=int,
        default=10,
        help="Output shape of the model",
    )

    args.add_argument(
        "--block",
        type=str,
        default="bottle_neck",
        help="Block to use for training",
    )

    args = args.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)
