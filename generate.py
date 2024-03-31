import pickle
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-w", "--weights", type=str, help="Path of weights created from train.py"
    )
    parser.add_argument(
        "-c", "--count", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output.txt", help="Output text file path"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=random.randint(0, 99999999), help="Seed value"
    )
    parser.add_argument(
        "-sw",
        "--start_with",
        type=str,
        default="",
        help="Start the predictions with given name",
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    # Load model
    with open(args.weights, "rb") as f:
        model = pickle.load(f)
    predictions = model.generate(
        count=args.count, seed=args.seed, start_with=args.start_with
    )
    # Write an output file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
    print(f"Ouput file: {args.output}")
