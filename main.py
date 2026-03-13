from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Starting InkOrPixel")

    train_model()
    evaluate_model()

if __name__ == "__main__":
    main()
