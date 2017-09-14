import gensim

# model = gensim.models.KeyedVectors.load_word2vec_format('./model/glove.6B.50d.txt', binary=False)  
model = gensim.models.KeyedVectors.load_word2vec_format('./model/glove.6B.300d.txt', binary=False)  
word_vectors = model.wv

prev = ''
while True:
	words = input("> ")
	words = words.split()
	if len(words) < 2: continue
	if words[0] != '-': words = ['+'] + words
	if len(words) % 2 != 0:
		print('wrong input')
		continue
	words = [words[i:i+2] for i in range(0, len(words), 2)]
	pos, neg = [], []
	for o, w in words:
		if 	o == '+': pos.append(w)
		elif 	o == '-': neg.append(w)
		else:
			print('wrong input')
			continue
	try:
		vects = word_vectors.most_similar(positive=pos, negative=neg)
		for vec in vects:
			print("%s (%2.0f)" % (vec[0], vec[1]*100), end=' ')
			print('\n')
	except: print('wrong input')
