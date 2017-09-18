import numpy as np
import tensorflow as tf
import keras
from keras import objectives, backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import gensim


nTest = 3

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


class PrintExample(keras.callbacks.Callback):
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

class VAE(object):
	def create(self, vocab_size=50, max_length=30, latent_rep_size=200):
		x = Input(shape=(max_length, vocab_size))
		vae_loss, encoded = self._build_encoder(x, latent_rep_size=latent_rep_size, max_length=max_length)

		self.encoder = Model(inputs=x, outputs=encoded)
		encoded_input = Input(shape=(latent_rep_size,))
		decoded = self._build_decoder(encoded_input, vocab_size, max_length)
		self.decoder = Model(encoded_input, decoded)

		self.autoencoder = Model(inputs=x, outputs=self._build_decoder(encoded, vocab_size, max_length))
		print(self.autoencoder.summary())
		self.autoencoder.compile(optimizer='Adam', loss=vae_loss, metrics=['accuracy'])

	def _build_encoder(self, x, latent_rep_size=200, max_length=30, epsilon_std=0.001):
# 		h = Bidirectional(LSTM(50, return_sequences=True, name='lstm_1'), merge_mode='concat')(x)
# 		h = Bidirectional(LSTM(50, return_sequences=False, name='lstm_2'), merge_mode='concat')(h)
		h = LSTM(500, return_sequences=True, name='lstm_1')(x)
		h = LSTM(500, return_sequences=False, name='lstm_2')(h)
		h = Dense(500, activation='relu', name='dense_1')(h)
		def sampling(args):
			z_mean_, z_log_var_ = args
			batch_size = K.shape(z_mean_)[0]
			epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
			return z_mean_ + K.exp(z_log_var_ / 2) * epsilon
		z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
		z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)
		def vae_loss(x, x_decoded_mean):
			x = K.flatten(x)
			x_decoded_mean = K.flatten(x_decoded_mean)
# 			xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
# 			xent_loss = max_length * objectives.categorical_crossentropy(x, x_decoded_mean)
			xent_loss = objectives.kullback_leibler_divergence(x, x_decoded_mean)
# 			xent_loss = cos_distance(x, x_decoded_mean)
			kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
			return xent_loss + kl_loss
		return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))
	
	def _build_decoder(self, encoded, vocab_size, max_length):
		repeated_context = RepeatVector(max_length)(encoded)
		h = LSTM(500, return_sequences=True, name='dec_lstm_1')(repeated_context)
		h = LSTM(500, return_sequences=True, name='dec_lstm_2')(h)
		return TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)

def create_callbacks(dir, log, model_name):
	import os
	import datetime
	dt = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

	checkpointPath = './%s/%s-%s-{epoch:05d}-{val_loss:.2f}.h5' % (dir, model_name, dt)
	checkpointDir = os.path.dirname(checkpointPath)
	try:
		os.stat(checkpointDir)
	except:
		os.mkdir(checkpointDir)

	logPath = './%s/%s_%s' % (log, model_name, dt)
	logDir = os.path.dirname(logPath)
	try:
		os.stat(logDir)
	except:
		os.mkdir(logDir)

	return	 	[	
			keras.callbacks.ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True),
			keras.callbacks.TensorBoard(log_dir=logPath),
# 			WriteExample(log_dir=logPath),
			PrintExample(),
			]

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

# data = np.load('./data/haiku.npy')
data = np.load('./data/sentences.npy')
print('Data shape: %s' % (str(data.shape)))
datLen  = data.shape[0]
sentLen = data.shape[1]
vecLen  = data.shape[2]

#data = data[:30]
testData = data[:nTest]
data = data[nTest:]

model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec/glove.6B.50d.txt', binary=False)  
wordVecs = model.wv
testSentences = array_to_sentences(testData)

# model = VAE()
# model.create(vocab_size=vecLen, max_length=sentLen)
# callbacks = create_callbacks('saves', 'logs', 'haiku_ae')
# model.autoencoder.fit(x=data, y=data, validation_split=0.2,
# 		batch_size=10, epochs=999, callbacks=callbacks)
# 
# exit(0)

import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers.convolutional import Cropping1D, UpSampling1D

inp = Input(shape=(sentLen, vecLen))
x = LSTM(50, return_sequences=True)(inp)
x = LSTM(50, return_sequences=True)(x)
x = LSTM(100, return_sequences=True)(x)
x = LSTM(225, return_sequences=False)(x)
x = keras.layers.core.RepeatVector(sentLen)(x)
x = LSTM(225, return_sequences=True)(x)
x = LSTM(100, return_sequences=True)(x)
x = LSTM(50, return_sequences=True)(x)
out = LSTM(vecLen, return_sequences=True)(x)

model = Model(inputs=inp, outputs=out)
model.compile(loss=cos_distance, optimizer='adam', metrics=['accuracy'])
print(model.summary())
		
callbacks = create_callbacks('saves', 'logs', 'haiku_ae')
model.fit(x=data, y=data, epochs=9999, validation_split=0.2, callbacks=callbacks)
