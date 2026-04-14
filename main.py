import argparse

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image


def main():
    parser = argparse.ArgumentParser(description="InkOrPixel")
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "predict"],
        help="Command to run"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for prediction"
    )

    args = parser.parse_args()

    if args.command == "train":
        print("Starting InkOrPixel")
        print("Training model...")
        train_model()

    elif args.command == "evaluate":
        print("Evaluating model...")
        evaluate_model()

    elif args.command == "predict":
        if not args.image:
            print("Please provide an image path with --image")
            return
        predict_image(args.image)


if __name__ == "__main__":
    main()