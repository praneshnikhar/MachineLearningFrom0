from src.generate_data import main as generate_main
from src.pipeline import run
from src.train import main as train_main


def demo():
    print("=" * 60)
    print("1. Generating synthetic transaction data")
    print("=" * 60)
    generate_main()

    print("\n" + "=" * 60)
    print("2. Training the classifier")
    print("=" * 60)
    train_main()

    print("\n" + "=" * 60)
    print("3. Running the pipeline on new (unlabeled) transactions")
    print("=" * 60)
    run()


if __name__ == "__main__":
    demo()
