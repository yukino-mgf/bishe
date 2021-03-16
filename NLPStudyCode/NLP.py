import torch
from fastNLP.embeddings import StaticEmbedding
from fastNLP import Vocabulary

vocab1 = Vocabulary()
vocab1.add_word_lst("this is a dema .".split())

embed1 = StaticEmbedding(vocab1, model_dir_or_name='en-glove-6b-50d')

words1 = torch.LongTensor([[vocab1.to_index(word) for word in "this is a dema .".split()]])
print(words1)
print(embed1(words1).size())
print(embed1(words1))

vocab2 = Vocabulary()
vocab2.add_word_lst("this is a demo .".split())
embed2 = StaticEmbedding(vocab2, model_dir_or_name='en-glove-6b-50d')
words2 = torch.LongTensor([[vocab2.to_index(word) for word in "this is a demo .".split()]])
print(words2)
print(embed2(words2).size())
print(embed2(words2))

print("result\n", embed1(words1)-embed2(words2))