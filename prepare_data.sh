#!/bin/bash

modelDir='word2vec'
dataDir='data'

## check dependencies
python3 -c 'import keras; import gensim' || exit 1

## word2vec
mkdir $modelDir
cd $modelDir

echo 'Downloading word2vec..'
wget http://nlp.stanford.edu/data/glove.6B.zip -q
unzip glove.6B.zip # uncompressing
# annotate vector files with number of words and vector length for gensim load
echo 'Annotating word2vec..'
sed -i '1s/^/400000 50\n/'   glove.6B.50d.txt
sed -i '1s/^/400000 100\n/' glove.6B.100d.txt
sed -i '1s/^/400000 200\n/' glove.6B.200d.txt
sed -i '1s/^/400000 300\n/' glove.6B.300d.txt

cd ..

## datasets
mkdir $dataDir
cd $dataDir

echo 'Downloading haiku..'
wget https://github.com/herval/haikuzao/raw/master/inputs/haiku.txt -q
sed -i 's/\([\.]\)/ \1 /g' haiku.txt # making punctuation readable

echo 'Downlaoding sentence corpus..'
wget http://academiccommons.columbia.edu/download/fedora_content/download/ac:175064/CONTENT/masc_word_sense_sentence_corpus.V1.0.tar.gz -q
tar -xvzf masc_word_sense_sentence_corpus.V1.0.tar.gz # uncompressing
awk 'BEGIN{FS="\t"} {print $7}' masc_word_sense_sentence_corpus.V1.0/masc_sentences.tsv | tr -d '.' > sentences.txt

echo 'Preprocessing haiku..'
python3 ../haiku_to_word2vec.py ../word2vec/glove.6B.50d.txt haiku.txt haiku.npy

echo 'Preprocessing sentences..'
python3 ../sentences_to_word2vec.py ../word2vec/glove.6B.50d.txt sentences.txt sentences.npy
