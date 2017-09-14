import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('./model/glove.6B.50d.txt', binary=False)  
wordVecs = model.wv

haikus, haiku = [], []
with open('haiku.txt') as file:
	for line in file:
		words = line.split()
		if len(words) == 0:
			haikus.append(haiku)
			haiku = []
			continue
		words = [word.lower() for word in words]
		vectors = []
		for word in words:
			try:
				vector = wordVecs[word]
			except:
				vector = wordVecs['@']
			vectors.append(vector)
		haiku += vectors
haikus = np.array(haikus)
print(haikus.shape)
np.save('haiku.npy', haikus)
