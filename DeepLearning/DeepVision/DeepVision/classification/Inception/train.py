import tensorflow as tf
import argparse
from DeepVision.classification.Inception.model import GoogLeNet
from DeepVision.utils.data import Datasets
from DeepVision.utils.helper import *
import ast


class TupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(ast.literal_eval(values)))

# make an instance of the LeNet class
def load_model(args):
    googlenet = GoogLeNet(input_shape=args.input_shape, output_shape=args.output_shape)

    model = googlenet.build(args.preprocess, args.naive, args.mid_output)

    return model


def main(args):
    model = load_model(args)

    # model_summary_only only
    if args.summary_only:
        model.summary()

    # if args.architecture:
    #     tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True)


    if args.train:

        dataset = Datasets().load_dataset(args.dataset)

        (x_train, y_train), (x_test, y_test) = dataset


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

    args.add_argument("--batch_size", type=int, default=256, help="Batch size")

    args.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

    args.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset to use for training"
    )

    args.add_argument(
        "--naive",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Whether to use naive or not",
    )

    args.add_argument(
        "--mid_output",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Whether to use mid_output or not",
    )
    

    args.add_argument(
        "--preprocess",
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

    # args.add_argument(
    #     "--input_shape",
    #     type=tuple,
    #     default=(224, 224, 3),
    #     help="Input shape of the model",
    # )

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

    args = args.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)
