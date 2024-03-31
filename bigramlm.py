import torch
import torch.nn.functional as F
import random


class BiGramLM(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.vocab_size = 0
        self.W = None
        self.itos = None
        self.stoi = None
        self.start_end_token = '.'
        self.max_rand_val = 99999999999

    def train(self, words, seed=None, learning_rate=0.1, trainig_loops=100):
        seed = seed or random.randint(0, self.max_rand_val)
        g = torch.Generator(device=self.device).manual_seed(seed)
        # Create vocabulary
        vocabulary = sorted(set(''.join(words)))
        vocabulary.insert(0, self.start_end_token)
        self.vocab_size = len(vocabulary)
        # Ordinal encode/decode
        self._generate_ordinal_encoder_decoder(vocabulary=vocabulary)
        # Make traing dataset
        x_train, y_train = self._make_training_dataset(words)
        # One Hot encode self.x_train
        x_enc = F.one_hot(x_train, num_classes=self.vocab_size).float().to(self.device)
        # Create Weights of size vocab_size with Neurons of vocab_size
        self.W = torch.randn((self.vocab_size, self.vocab_size),
                             requires_grad=True,
                             generator=g,
                             device=self.device)
        self.W.retain_grad()
        # Training loop
        for _ in range(trainig_loops):
            # Forward pass
            logits = x_enc @ self.W
            counts = logits.exp()
            probs = counts / counts.sum(dim=-1, keepdim=True)
            loss = -probs[torch.arange(len(x_train)), y_train].log().mean()
            # Backward pass
            self.W.grad = None
            loss.backward()
            # Update Weights
            self.W.data -= learning_rate * self.W.grad
        print(f'Training loss: {loss.item()}')

    def _generate_ordinal_encoder_decoder(self, vocabulary):
        """Generate the ordinal ecoder/decorder from the
        given vocabulary, Vocabulary should be shorted before
        passing it"""
        self.itos = dict(enumerate(vocabulary))
        self.stoi = dict((s, i) for i, s in self.itos.items())

    def _make_training_dataset(self, words):
        # Make training set using bigram
        x_train = []
        y_train = []
        for word in words:
            word = f'{self.start_end_token}{word}{self.start_end_token}'
            for ch1, ch2 in zip(word, word[1:]):
                x_train.append(self.stoi[ch1])
                y_train.append(self.stoi[ch2])
        return torch.tensor(x_train), torch.tensor(y_train)

    def generate(self, seed=None, count=10, start_with=''):
        seed = seed or random.randint(0, self.max_rand_val)
        predictions = []
        g = torch.Generator().manual_seed(seed)

        while len(predictions) < count:
            idx = self.stoi[(self.start_end_token + start_with)[-1]]
            predict = []
            while True:
                x_enc = F.one_hot(torch.tensor(idx), num_classes=self.vocab_size).float()
                logits = x_enc @ self.W.data
                counts = logits.exp()
                probs = counts / counts.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                if not idx:
                    break
                predict.append(self.itos[idx])

            predict = start_with + ''.join(predict)
            if predict == start_with or predict in predictions:
                continue
            predictions.append(predict)
        return predictions