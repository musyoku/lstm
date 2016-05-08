# -*- coding: utf-8 -*-
import os, sys, time, codecs
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import vocab
from args import args
from env import dataset, n_vocab, n_dataset, lstm, conf

# Windowsでprintする用
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors="xmlcharrefreplace")

# 学習時に長さ制限した場合は同じ値をここにもセット
current_length_limit = 15

def get_validation_data():
	max_length_in_batch = 0
	length = current_length_limit + 1
	while length > current_length_limit:
		k = np.random.randint(0, n_dataset)
		data = dataset[k]
		length = len(data)
	return data

for phrase in xrange(100):
	lstm.reset_state()
	str = ""
	char = get_validation_data()[0]
	for n in xrange(100):
		str += vocab.id_to_word(char)
		id = lstm.predict(char, test=True)
		if id == 0:
			break
		char = id
	print str

