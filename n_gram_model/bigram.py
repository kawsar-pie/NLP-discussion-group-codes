from n_gram import NGramLanguageModel

bigram_model = NGramLanguageModel(n=2)
corpus = ["My name is Kawsar", "I am a very good boy",
          "but my mother says I am a very bad boy", "What can I do now?"]
bigram_model.train(corpus)
# returns a log probability
score = bigram_model.score("I am a very good")
print(score)

