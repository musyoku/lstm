import codecs, random
import numpy as np

ID_EOS = 0

def read_data(filename_train=None, filename_dev=None, filename_test=None):
	vocab_str_id = {
		"<eos>": ID_EOS,
	}
	dataset_train = []
	dataset_dev = []
	dataset_test = []

	if filename_train is not None:
		with codecs.open(filename_train, "r", "utf-8") as f:
			dataset_train.append(ID_EOS)
			for sentence in f:
				sentence = sentence.strip()
				if len(sentence) == 0:
					continue
				words = sentence.split(" ")
				for word in words:
					if word not in vocab_str_id:
						vocab_str_id[word] = len(vocab_str_id)
					word_id = vocab_str_id[word]
					dataset_train.append(word_id)
				dataset_train.append(ID_EOS)

	if filename_dev is not None:
		with codecs.open(filename_dev, "r", "utf-8") as f:
			dataset_dev.append(ID_EOS)
			for sentence in f:
				sentence = sentence.strip()
				if len(sentence) == 0:
					continue
				words = sentence.split(" ")
				for word in words:
					if word not in vocab_str_id:
						vocab_str_id[word] = len(vocab_str_id)
					word_id = vocab_str_id[word]
					dataset_dev.append(word_id)
				dataset_dev.append(ID_EOS)

	if filename_test is not None:
		with codecs.open(filename_test, "r", "utf-8") as f:
			dataset_test.append(ID_EOS)
			for sentence in f:
				sentence = sentence.strip()
				if len(sentence) == 0:
					continue
				dataset_test = [ID_BOS]
				words = sentence.split(" ")
				for word in words:
					if word not in vocab_str_id:
						vocab_str_id[word] = len(vocab_str_id)
					word_id = vocab_str_id[word]
					dataset_test.append(word_id)
				dataset_test.append(ID_EOS)

	vocab_id_str = {}
	for word, word_id in vocab_str_id.items():
		vocab_id_str[word_id] = word

	return dataset_train, dataset_dev, dataset_test, vocab_str_id, vocab_id_str
