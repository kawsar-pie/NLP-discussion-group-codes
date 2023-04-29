from collections import defaultdict
import math
class NGramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(int)
        self.context_counts = defaultdict(int)

    def train(self, corpus):
        for sentence in corpus:
            tokens = sentence.split()
            # print("TOKENS ARE:")
            # print(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                # print(str(self.n)+"-GRAMS FOR THE TOKEN:")
                # print(ngram)
                context = ngram[:-1]
                # print("CONTEXT:")
                # print(context)
                self.counts[ngram] += 1
                self.context_counts[context] += 1
        print(self.counts)
        print(self.context_counts)

    def score(self, sentence):
        tokens = sentence.split()
        logprob = 0.0
        for i in range(self.n-1, len(tokens)):
            ngram = tuple(tokens[i-self.n+1:i+1])
            context = ngram[:-1]
            count = self.counts[ngram]
            context_count = self.context_counts[context]
            if count == 0 or context_count == 0:
                logprob += float('-inf')
            else:
                probability = count / context_count
                logprob += math.log(probability)
        return logprob

    