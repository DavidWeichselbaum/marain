#!/bin/bash

modelDir='word2vec'
dataDir='data'

## word2vec
mkdir $modelDir
cd $modelDir

echo 'Downloading word2vec..'
wget http://nlp.stanford.edu/data/glove.6B.zip -q --show-progress
unzip glove.6B.zip # uncompressing
# annotate vector files with number of words and vector length for gensim load
echo 'Annotating word2vec..'
sed -i '1s/^/400000 50\n/'   glove.6B.50d.txt
sed -i '1s/^/400000 100\n/' glove.6B.100d.txt
sed -i '1s/^/400000 200\n/' glove.6B.200d.txt
sed -i '1s/^/400000 300\n/' glove.6B.300d.txt

## datasets
mkdir $dataDir
cd $dataDir

echo 'Downloading haiku..'
wget https://github.com/herval/haikuzao/raw/master/inputs/haiku.txt -q --show-progress
sed -i '.bak' 's/\([\.]\)/ \1 /g' haiku.txt # making punctuation readable

echo 'Downlaoding sentence corpus..'
wget http://academiccommons.columbia.edu/download/fedora_content/download/ac:175064/CONTENT/masc_word_sense_sentence_corpus.V1.0.tar.gz -q --show-progress
tar -xvzf masc_word_sense_sentence_corpus.V1.0.tar.gz # uncompressing

echo 'Preprocessing haiku..'
python3 ../haiku_to_word2vec.py

echo 'Preprocessing sentences..'
bash ../masc_extractor.sh 
ptyhon3 ../sentences_to_word2vec.py 
