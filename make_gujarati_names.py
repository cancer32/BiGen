import torch
import argparse
import random

if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='PyTorch Template')
    argparse.add_argument('-c', '--count',  type=int, default=10, help='Number of samples to generate')
    argparse.add_argument('-g', '--gender', type=str, default='male', help='Gender (male/female)')
    argparse.add_argument('-o', '--output', type=str, default='output.txt', help='Output text file path')
    argparse.add_argument('-s', '--seed', type=int, default=random.randint(0, 99999999), help='Seed value')
    argparse.add_argument('-sw', '--start_with', type=str, default='', help='Start the predictions with given name')
    args = argparse.parse_args()
    print(f'Args: {args}')

    seed = args.seed
    count = args.count
    gender = args.gender
    output = args.output
    start_with = args.start_with

    # Read dataset
    data_set = f'names_{gender}.txt'
    print(f'Reading {data_set}...')
    with open(data_set, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()

    # Make the vocabulary from the dataset
    vocabulary = sorted(set(''.join(words)))
    vocabulary.insert(0, '.')
    vocab_size = len(vocabulary)
    print(f'Vocabulary size: {vocab_size}')
    stoi = dict((s, i) for i, s in enumerate(vocabulary))
    itos = dict((i, s) for s, i in stoi.items())

    # Generate bigram model
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for word in words:
        word = f'.{word}.'
        for ch1, ch2 in zip(word, word[1:]):
            idx1, idx2 = stoi[ch1], stoi[ch2]
            N[idx1, idx2] += 1
    P = N / N.sum(1, keepdim=True)

    # Make predictions
    predictions = []
    g = torch.Generator().manual_seed(seed)
    while len(predictions) < count:
        idx = stoi[('.' + start_with)[-1]]
        predict = []
        while True:
            idx = torch.multinomial(P[idx], num_samples=1, replacement=True, generator=g).item()
            if not idx:
                break
            predict.append(itos[idx])
        predict = (start_with + ''.join(predict)).replace('.', '')
        if predict == start_with or predict in predictions:
            continue
        predictions.append(predict)

    # Write an output file
    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(predictions))
    print(f'Ouput file: {output}')