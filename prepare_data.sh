#!/bin/bash

modelDir='word2vec_tmp'
dataDir='data_tmp'

## word2vec
mkdir $modelDir
cd $modelDir

echo 'Downloading word2vec..'
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip # uncompress
# annotate vector files with number of words and vector length for gensim load
sed -i '1s/^/400000 50\n/'   glove.6B.50d.txt
sed -i '1s/^/400000 100\n/' glove.6B.100d.txt
sed -i '1s/^/400000 200\n/' glove.6B.200d.txt
sed -i '1s/^/400000 300\n/' glove.6B.300d.txt

## datasets
mkdir $dataDir
cd $dataDir

echo 'Downloading haiku..'
wget https://github.com/herval/haikuzao/raw/master/inputs/haiku.txt
sed -i '.bak' 's/\([\.]\)/ \1 /g' haiku.txt # making punctuation readable

echo 'Downlaoding sentence corpus..'
wget http://academiccommons.columbia.edu/download/fedora_content/download/ac:175064/CONTENT/masc_word_sense_sentence_corpus.V1.0.tar.gz
tar -xvzf masc_word_sense_sentence_corpus.V1.0.tar.gz # uncompressing

echo 'Preprocessing haiku..'
python3 ../haiku_to_word2vec.py

echo 'Preprocessing sentences..'
ptyhon3 ../masc_extractor.sh 
