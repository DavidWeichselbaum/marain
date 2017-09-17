#!/bin/bash

awk 'BEGIN{FS="\t"} {print $7}' masc_word_sense_sentence_corpus.V1.0/masc_sentences.tsv | tr -d '.'
