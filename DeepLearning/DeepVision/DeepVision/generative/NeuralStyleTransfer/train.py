from DeepVision.generative.NeuralStyleTransfer.model import NeuralStyleTransfer
import argparse





def load_model(args):
    model = NeuralStyleTransfer(
        style_image=args.style_image,
        content_image=args.content_image,
        extractor=args.extractor,
        n_style_layers=args.n_style_layers,
        n_content_layers=args.n_content_layers,
        display=args.display,
    )

    return model

def train(model, args):
    all_images = model.fit_style_transfer(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        steps_per_epoch=args.steps_per_epoch,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        show_image=args.show_image,
        show_interval=args.show_interval,
        var_weight=args.var_weight,
        terminal=args.terminal,
    )

    return all_images


def main(args):
    # load the model
    model = load_model(args)

    # train the model
    all_images = train(model, args)

    # save the animation
    if args.animate:
        model.make_annime(all_images, args.animation_name, args.duration)





def arg_parse():
    args = argparse.ArgumentParser(add_help=True)

    args.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args.add_argument(
        "--steps_per_epoch", type=int, default=100, help="Steps per epoch"
    )

    args.add_argument("--learning_rate", type=float, default=80., help="Initial Learning rate")

    args.add_argument("--style_weight", type=float, default=1e-4, help="Style weight")

    args.add_argument(
        "--content_weight", type=float, default=1e4, help="Content weight"
    )

    args.add_argument(
        "--show_image",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Whether to show image or not",
    )

    args.add_argument(
        "--show_interval",
        type=int,
        default=10,
        help="Interval to show image",
    )

    args.add_argument(
        "--var_weight",
        type=float,
        default=1e-4,
        help="Weight for total variation loss",
    )

    args.add_argument(
        "--terminal",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="What are you using to run this code, terminal or jupyter notebook",
    )

    args.add_argument(
        "--style_image",
        type=str,
        default="style.jpg",
        help="Style image to use for training",
    )

    args.add_argument(
        "--content_image",
        type=str,
        default="content.jpg",
        help="Content image to use for training",
    )

    args.add_argument(
        "--extractor",
        type=str,
        default="inception_v3", 
        help="Extractor to use for training, only inception_v3 is supported or give the instance of the extractor"
    )

    args.add_argument(
        "--n_style_layers",
        type=int,
        default=5,
        help="Number of style layers to use for training",
    )

    args.add_argument(
        "--n_content_layers",
        type=int,
        default=1,
        help="Number of content layers to use for training",
    )

    args.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to display input style and content images or not",
    )

    args.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to animate the training or not",
    )

    args.add_argument(
        "--animation_name",
        type=str,
        default="animation.gif",
        help="Name of the animation",
    )

    args.add_argument(
        "--duration",
        type=int,
        default=2,
        help="Duration of the animation",
    )

    args = args.parse_args()

    return args



if __name__ == "__main__":
    args = arg_parse()
    main(args)
