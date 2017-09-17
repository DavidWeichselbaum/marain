# Marain

In his 'Culture' novel series, Ian M. Banks describes a language called Marain. 
In Bank's books, both the spoken and written aspect of Marain to have been developed by artificial intelligences to enable efficient and lossless communication between both humans and machines.

This repository is an attempt of implementing the visual aspect of Marain, (called Diaglyphs or Glyphs).
Glyphs are described to being able to hold any amount of conversational data, ranging from single vowels to whole books including encoded metadata.
This is achieved by the fractal nature of glyphs -- having a quick glance at a glyph holding a paragraph will tell you it's synopsis, studying it in details reveals the full content including pronunciation, capitalization and connotation.

I try to solve the problem of encoding text in a scalable, human-recognizable glyph including visual error-correction.
This is done by using a recurrent neural network autoencoder, with the encoded vector being represented as a grayscale image.
The image is decoded via a convolutional neural networks based on frozen weigths stemming from an image recognition task. 
Since image recognition involves understanding objects in a similar way than humans do, this should ensure that glyphs are human-readable.

Here, a variational autencoder is used, based on [George A. Adam's](https://github.com/georgeadam) excellent [blog post](http://alexadam.ca/ml/2017/05/05/keras-vae.html) and adapted to handle not 1hot but [gensim word2vec](https://radimrehurek.com/gensim/models/word2vec.html) encoding.
The word2vec variant used in this project encodes 400,000 words to a vector of 50 real numbers, was trained on Wikipedia and Gigaword 5. 
It can be found [here](https://github.com/3Top/word2vec-api).

The visual aspect of this project is not yet engaged, since i want to start from a decent latent space representation of text.
Also, I'm currently fond of haikus as a [training set](https://github.com/herval/haikuzao/blob/master/inputs/haiku.txt) for the autoencoder, since they are short and similarly formed in general, but in themselves already a compressed form of expression.
Alternatively you can train it on the [MASC word sense sentence corpus](https://www.cs.vassar.edu/~ide/papers/masc-collab-wordsense.pdf) by changing the relevant line in marain.py.

To download an preprocess the data, do:

> bash prepare_data.sh

This may take a while -- word2vec is huge!

Then simply run:

> python3 marain.py

to start the training of the autoencoder.

Once started, you can monitor the training progress and sample text outputs with tensorboard:

> tensorboard --logdir ./logs

