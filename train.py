"""Training module for a character level bigram language model"""

import random
import pickle
from bigramlm import BigramLM

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", required=True, help="An input dataset text file path"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Ouput path of the weights"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=random.randint(0, 99999999), help="Seed value"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=1, help="Learing rate"
    )
    parser.add_argument(
        "-tl", "--training_loop", type=int, default=10, help="Number of training loops"
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    # Read the dataset
    with open(args.input, "r", encoding="utf-8") as f:
        words = f.read().splitlines()

    # Create model
    model = BigramLM()
    # Provide training dataset to Train the model
    model.train(
        words=words,
        seed=args.seed,
        learning_rate=args.learning_rate,
        trainig_loops=args.training_loop,
    )
    # Write model weights
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
