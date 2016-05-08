# -*- coding: utf-8 -*-
import os
import numpy as np
import codecs

vocab = {}
inv_vocab = {}

def load(dir):
	fs = os.listdir(dir)
	print "loading", len(fs), "files..."
	dataset = []
	vocab["<eos>"] = 0
	for fn in fs:
		unko = codecs.open("%s/%s" % (dir, fn), "r", "utf_8_sig")	# BOMありならutf_8_sig　そうでないならutf_8
		for line in unko:
			line = line.replace("\n", "")
			data = np.empty((len(line) + 1,), dtype=np.int32)
			for i in xrange(len(line)):
				word = line[i]
				if word not in vocab:
					vocab[word] = len(vocab)
					inv_vocab[vocab[word]] = word
				data[i] = vocab[word]
			data[len(line)] = 0
			dataset.append(data)
	n_vocab = len(vocab)
	n_dataset = len(dataset)
	print "# of chars:", n_vocab
	print "# of data:", n_dataset
	return dataset, n_vocab, n_dataset

def id_to_word(id):
	return inv_vocab[id]

def word_to_id(word):
	return vocab[word]