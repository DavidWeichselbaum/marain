import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('./model/glove.6B.50d.txt', binary=False)  
wordVecs = model.wv

maxLen = 30
unknown = wordVecs['@']
spacer = wordVecs['$']

sentences = []
with open('./data/sentences.txt') as file:
	for i, line in enumerate(file):
		words = line.split()
		if len(words) > 30: continue
		words = [word.lower() for word in words]
		vectors = []
		for word in words:
			try:
				vector = wordVecs[word]
			except:
				vector = unknown
			vectors.append(vector)
		sentences.append(vectors)
		print('Converting sentence: %d' % (i), end='\r')
sentences = np.array(sentences)

datLen = len(sentences)
print('Data lenght: %d' % (datLen))
vecLen = sentences[0][0].shape[0]
print('Vector lenght: %d' % (vecLen))
sentLen = 0
for sent in sentences:
	if len(sent) > sentLen: sentLen = len(sent)
print('Sentence lenght: %d' % (sentLen))

data = np.tile(spacer, (datLen, sentLen, 1))
print('Data shape: %s' % (str(data.shape)))
for i, sent in enumerate(sentences):
	sent = np.array(sent)
	if len(sent) == 0: continue
	data[i, :len(sent), :] = sent

print('Saving data.')
np.save('./data/sentences.npy', data)
