import numpy as np
import tensorflow as tf
import gensim

nTest = 3

rawData = np.load('haiku.npy')
datLen = len(rawData)
print('Data lenght: %d' % (datLen))
vecLen = rawData[0][0].shape[0]
print('Vector lenght: %d' % (vecLen))
sentLen = 0
for sent in rawData:
	if len(sent) > sentLen: sentLen = len(sent)
print('Sentence lenght: %d' % (sentLen))

model = gensim.models.KeyedVectors.load_word2vec_format('./model/glove.6B.50d.txt', binary=False)  
wordVecs = model.wv
spacer = wordVecs['$']
data = np.tile(spacer, (datLen, sentLen, 1))
print('Data shape: %s' % (str(data.shape)))
for i, sent in enumerate(rawData):
	sent = np.array(sent)
	if len(sent) == 0: continue
	data[i, :len(sent), :] = sent

#data = data[:30]
testData = data[:nTest]
data = data[nTest:]

def array_to_sentences(array):
	sentences = []
	for vectors in array:
		sentence = []
		for vector in vectors:
			word = wordVecs.most_similar(positive=[vector], topn=1)[0][0]
			if word == '$': continue
			sentence.append(word)
		sentence = ' '.join(sentence)
		sentences.append(sentence)
	return sentences

testSentences = array_to_sentences(testData)

import keras
class WriteExample(keras.callbacks.Callback):
	def __init__(self, log_dir='./logs'):
		self.logdir = log_dir
		with tf.name_scope('validation_sentences') as scope:
			self.id_list = tf.placeholder(tf.int32, shape=[nTest*2], name='sent_ids')
			self.valid_placeholder = tf.placeholder(tf.string, name='valid_summaries')
			self.summary = tf.summary.text('original_decoded', self.valid_placeholder)
		self.sess = tf.Session()
		self.summary_writer = tf.summary.FileWriter(logdir=self.logdir, graph=self.sess.graph)
	def on_epoch_end(self, epoch, logs={}):
		if epoch % 1 != 0: return
	
		decoded = self.model.predict(testData)
		decSentences = array_to_sentences(decoded)

		textList = []
		for original, decoded in zip(testSentences, decSentences):
			textList += ['O: ' + original]
			textList += ['D: ' + decoded]	
		print()
		print('\n'.join(textList))
		id2sent = {id:sent for id, sent in enumerate(textList)}
		sent2id = {sent:id for id, sent in id2sent.items()}
		
		predicted_sents_ids = self.sess.run(
			self.id_list,
			feed_dict={
				self.id_list: list(range(nTest*2))
			})
		predicted_sents = [id2sent[id] for id in predicted_sents_ids]
		valid_summary = self.sess.run(self.summary, feed_dict={
			self.valid_placeholder: predicted_sents
		})
		self.summary_writer.add_summary(valid_summary, global_step=epoch)
		self.summary_writer.flush()

def cos_distance(y_true, y_pred):
	y_true = K.l2_normalize(y_true, axis=-1)
	y_pred = K.l2_normalize(y_pred, axis=-1)
	return K.mean(1 - K.sum((y_true * y_pred), axis=-1))

import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers.convolutional import Cropping1D, UpSampling1D

inp = Input(shape=(sentLen, vecLen))
x = LSTM(50, return_sequences=False)(inp)
# x = LSTM(50, return_sequences=True)(inp)
# x = LSTM(50, return_sequences=True)(x)
# x = LSTM(500, return_sequences=False)(x)
x = keras.layers.core.RepeatVector(sentLen)(x)
# x = LSTM(50, return_sequences=True)(x)
# x = LSTM(50, return_sequences=True)(x)
out = LSTM(vecLen, return_sequences=True)(x)

model = Model(inputs=inp, outputs=out)
model.compile(loss=cos_distance, optimizer='adagrad')
print(model.summary())

import datetime
dt = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
logDir = './logs/%s' % (dt)
callbacks = 	[	
		keras.callbacks.TensorBoard(log_dir=logDir),
		WriteExample(log_dir=logDir),
		]
		
model.fit(x=data, y=data, epochs=9999, validation_split=0.2, callbacks=callbacks)
