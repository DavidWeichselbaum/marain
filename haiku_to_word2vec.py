import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('./model/glove.6B.50d.txt', binary=False)  
wordVecs = model.wv

maxLen = 15
unknown = wordVecs['@']
spacer = wordVecs['$']
lineBreak = wordVecs['/']

haikus, haiku = [], []
with open('./data/haiku.txt') as file:
	for i, line in enumerate(file):
		words = line.split()
		if len(words) == 0:
			if len(haiku) > 0: del haiku[-1] # remove lineBreak from last line
			if len(haiku) <= maxLen: haikus.append(haiku)
			haiku = []
			continue

		words = [word.lower() for word in words]
		vectors = []
		for word in words:
			try:
				vector = wordVecs[word]
			except:
				vector = unknown
			vectors.append(vector)
		haiku += vectors
		haiku += [lineBreak]
		print('Converting haiku: %d' % (i), end='\r')
haikus = np.array(haikus)

datLen = len(haikus)
print('Data lenght: %d' % (datLen))
vecLen = haikus[0][0].shape[0]
print('Vector lenght: %d' % (vecLen))
sentLen = 0
for sent in haikus:
	if len(sent) > sentLen: sentLen = len(sent)
print('Haiku lenght: %d' % (sentLen))

data = np.tile(spacer, (datLen, sentLen, 1))
print('Data shape: %s' % (str(data.shape)))
for i, sent in enumerate(haikus):
	sent = np.array(sent)
	if len(sent) == 0: continue
	data[i, :len(sent), :] = sent

print('Saving data.')
np.save('./data/haiku.npy', data)
