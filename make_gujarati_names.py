import torch
import argparse
import random

if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='PyTorch Template')
    argparse.add_argument('-c', '--count',  type=int, default=10, help='Number of samples to generate')
    argparse.add_argument('-g', '--gender', type=str, default='male', help='Gender (male/female)')
    argparse.add_argument('-o', '--output', type=str, default='output.txt', help='Output text file path')
    args = argparse.parse_args()

    seed = random.randint(0, 99999999)
    count = args.count
    gender = args.gender
    output = args.output
    start_from = ''

    with open(f'names_{gender}.txt', 'r',
               encoding='utf-8') as f:
        words = f.read().splitlines()

    vocabulary = sorted(set(''.join(words)))
    vocabulary.insert(0, '.')
    vocab_size = len(vocabulary)

    stoi = dict((s, i) for i, s in enumerate(vocabulary))
    itos = dict((i, s) for s, i in stoi.items())

    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for word in words:
        word = f'.{word}.'
        for ch1, ch2 in zip(word, word[1:]):
            idx1, idx2 = stoi[ch1], stoi[ch2]
            N[idx1, idx2] += 1
    P = N / N.sum(1, keepdim=True)

    names = []
    g = torch.Generator().manual_seed(seed)
    while len(names) < count:
        idx = stoi[('.' + start_from)[-1]]
        predict = []
        while True:
            idx = torch.multinomial(P[idx], num_samples=1, replacement=True, generator=g).item()
            if not idx:
                break
            predict.append(itos[idx])
        predict = (start_from + ''.join(predict)).replace('.', '')
        if predict == start_from or predict in names:
            continue
        names.append(predict)

    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(names))